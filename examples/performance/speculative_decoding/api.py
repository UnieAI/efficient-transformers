# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
OpenAI-Compatible API Server with Request Scheduler for Speculative Decoding Inference
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# OpenAI API Compatible Models
# =============================================================================


class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author (system, user, assistant)")
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = Field(None, description="Optional name for the participant")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model ID to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=10)
    stream: Optional[bool] = Field(False)
    stop: Optional[Union[str, List[str]]] = Field(None)
    max_tokens: Optional[int] = Field(None)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    user: Optional[str] = Field(None)
    num_speculative_tokens: Optional[int] = Field(None)
    priority: Optional[int] = Field(0, description="Request priority (higher = more priority)")


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# =============================================================================
# Request and Scheduling Data Structures
# =============================================================================


class RequestStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class InferenceRequest:
    """Represents a single inference request"""
    request_id: str
    prompt: str
    messages: List[ChatMessage]
    max_tokens: int
    temperature: float
    top_p: float
    num_speculative_tokens: int
    priority: int
    created_at: float
    is_streaming: bool = False
    status: RequestStatus = RequestStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    # Use default=None and create in __post_init__ to avoid sharing between instances
    completion_event: Optional[asyncio.Event] = field(default=None, repr=False)
    # Queue for streaming tokens
    token_queue: Optional[asyncio.Queue] = field(default=None, repr=False)
    
    def __post_init__(self):
        if self.completion_event is None:
            self.completion_event = asyncio.Event()
        if self.is_streaming and self.token_queue is None:
            self.token_queue = asyncio.Queue()
    
    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


# =============================================================================
# Scheduler Implementation
# =============================================================================


class SchedulerStrategy(ABC):
    @abstractmethod
    async def add_request(self, request: InferenceRequest):
        pass
    
    @abstractmethod
    async def get_next_batch(self, max_batch_size: int) -> Optional[List[InferenceRequest]]:
        pass
    
    @abstractmethod
    def pending_count(self) -> int:
        pass


class FIFOScheduler(SchedulerStrategy):
    def __init__(self):
        self.queue: deque = deque()
        self._lock = asyncio.Lock()
    
    async def add_request(self, request: InferenceRequest):
        async with self._lock:
            self.queue.append(request)
    
    async def get_next_batch(self, max_batch_size: int) -> Optional[List[InferenceRequest]]:
        async with self._lock:
            if not self.queue:
                return None
            batch = []
            for _ in range(min(max_batch_size, len(self.queue))):
                batch.append(self.queue.popleft())
            return batch
    
    def pending_count(self) -> int:
        return len(self.queue)


class PriorityScheduler(SchedulerStrategy):
    def __init__(self):
        self.queue: List[InferenceRequest] = []
        self._lock = asyncio.Lock()
    
    async def add_request(self, request: InferenceRequest):
        async with self._lock:
            self.queue.append(request)
            self.queue.sort()
    
    async def get_next_batch(self, max_batch_size: int) -> Optional[List[InferenceRequest]]:
        async with self._lock:
            if not self.queue:
                return None
            batch = self.queue[:max_batch_size]
            self.queue = self.queue[max_batch_size:]
            return batch
    
    def pending_count(self) -> int:
        return len(self.queue)


class DynamicBatchScheduler(SchedulerStrategy):
    def __init__(self, length_buckets: List[int] = None):
        self.length_buckets = length_buckets or [128, 256, 512, 1024, 2048]
        self.buckets: Dict[int, List[InferenceRequest]] = {b: [] for b in self.length_buckets}
        self._lock = asyncio.Lock()
    
    def _get_bucket(self, prompt_length: int) -> int:
        for bucket in self.length_buckets:
            if prompt_length <= bucket:
                return bucket
        return self.length_buckets[-1]
    
    async def add_request(self, request: InferenceRequest):
        async with self._lock:
            bucket = self._get_bucket(len(request.prompt))
            self.buckets[bucket].append(request)
            self.buckets[bucket].sort()
    
    async def get_next_batch(self, max_batch_size: int) -> Optional[List[InferenceRequest]]:
        async with self._lock:
            best_bucket = None
            max_count = 0
            for bucket, requests in self.buckets.items():
                if len(requests) > max_count:
                    max_count = len(requests)
                    best_bucket = bucket
            
            if best_bucket is None or max_count == 0:
                return None
            
            batch = self.buckets[best_bucket][:max_batch_size]
            self.buckets[best_bucket] = self.buckets[best_bucket][max_batch_size:]
            return batch
    
    def pending_count(self) -> int:
        return sum(len(reqs) for reqs in self.buckets.values())


class RequestScheduler:
    """Main scheduler with event-driven batch processing"""
    
    def __init__(
        self,
        strategy: str = "fifo",
        max_batch_size: int = 8,
        max_wait_time: float = 0.05,  # Reduced wait time for faster response
        max_queue_size: int = 1000,
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_queue_size = max_queue_size
        
        if strategy == "fifo":
            self.strategy = FIFOScheduler()
        elif strategy == "priority":
            self.strategy = PriorityScheduler()
        elif strategy == "dynamic":
            self.strategy = DynamicBatchScheduler()
        else:
            raise ValueError(f"Unknown scheduling strategy: {strategy}")
        
        self._running = False
        self._process_task: Optional[asyncio.Task] = None
        self._inference_engine: Optional['InferenceEngine'] = None
        
        # Event to signal new request arrival - wakes up the processor immediately
        self._new_request_event = asyncio.Event()
        
        # Metrics
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.total_batches = 0
    
    def set_inference_engine(self, engine: 'InferenceEngine'):
        self._inference_engine = engine
    
    async def start(self):
        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info("Scheduler started")
    
    async def stop(self):
        self._running = False
        self._new_request_event.set()  # Wake up the loop
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")
    
    async def submit_request(self, request: InferenceRequest) -> str:
        if self.strategy.pending_count() >= self.max_queue_size:
            raise HTTPException(status_code=503, detail="Server is overloaded")
        
        await self.strategy.add_request(request)
        self.total_requests += 1
        
        # Signal the processor that a new request is available
        self._new_request_event.set()
        
        logger.debug(f"Request {request.request_id} submitted, pending: {self.strategy.pending_count()}")
        return request.request_id
    
    async def wait_for_result(self, request: InferenceRequest, timeout: float = 300.0) -> Dict:
        try:
            await asyncio.wait_for(request.completion_event.wait(), timeout=timeout)
            if request.status == RequestStatus.FAILED:
                raise HTTPException(status_code=500, detail=request.error or "Inference failed")
            return request.result
        except asyncio.TimeoutError:
            request.status = RequestStatus.FAILED
            raise HTTPException(status_code=504, detail="Request timeout")
    
    async def _process_loop(self):
        """Main processing loop - event driven"""
        while self._running:
            try:
                # Wait for new request or timeout
                try:
                    await asyncio.wait_for(
                        self._new_request_event.wait(),
                        timeout=self.max_wait_time
                    )
                except asyncio.TimeoutError:
                    pass
                
                # Clear the event
                self._new_request_event.clear()
                
                # Check if there are pending requests
                if self.strategy.pending_count() == 0:
                    continue
                
                # Optional: wait a bit more to collect more requests for batching
                # Only if we don't have enough for a full batch
                if self.strategy.pending_count() < self.max_batch_size:
                    await asyncio.sleep(self.max_wait_time)
                
                # Get next batch
                batch = await self.strategy.get_next_batch(self.max_batch_size)
                if batch is None or len(batch) == 0:
                    continue
                
                logger.info(f"Processing batch of {len(batch)} requests")
                
                # Process batch
                await self._process_batch(batch)
                self.total_batches += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in scheduler loop: {e}")
    
    async def _process_batch(self, batch: List[InferenceRequest]):
        if self._inference_engine is None:
            for req in batch:
                req.status = RequestStatus.FAILED
                req.error = "Inference engine not initialized"
                req.completion_event.set()
            return
        
        # Mark all requests as processing
        for req in batch:
            req.status = RequestStatus.PROCESSING
        
        try:
            # Run inference
            results = await self._inference_engine.run_batch(batch)
            
            # Update request results and notify waiters
            for req, result in zip(batch, results):
                req.result = result
                req.status = RequestStatus.COMPLETED
                
                # For streaming requests, put tokens in queue then signal done
                if req.is_streaming and req.token_queue is not None:
                    for token_id in result.get("token_ids", []):
                        await req.token_queue.put(("token", token_id))
                    await req.token_queue.put(("done", None))
                
                req.completion_event.set()
                self.completed_requests += 1
                
            logger.debug(f"Batch completed, {self.completed_requests} total completed")
                
        except Exception as e:
            logger.exception(f"Batch processing failed: {e}")
            for req in batch:
                req.status = RequestStatus.FAILED
                req.error = str(e)
                if req.is_streaming and req.token_queue is not None:
                    await req.token_queue.put(("error", str(e)))
                req.completion_event.set()
                self.failed_requests += 1
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "total_batches": self.total_batches,
            "pending_requests": self.strategy.pending_count(),
            "avg_batch_size": self.completed_requests / max(1, self.total_batches),
        }


# =============================================================================
# Inference Engine
# =============================================================================


class InferenceEngine:
    """Inference engine - runs in thread pool to not block event loop"""
    
    def __init__(
        self,
        target_model_name: str,
        prefill_seq_len: int = 256,
        ctx_len: int = 1024,
        num_speculative_tokens: int = 3,
        max_ngram_size: int = 3,
        device_group: List[int] = None,
        full_batch_size: int = 8,
        prefill_bsz: int = 1,
        num_sessions: int = 1,
    ):
        self.target_model_name = target_model_name
        self.prefill_seq_len = prefill_seq_len
        self.ctx_len = ctx_len
        self.num_speculative_tokens = num_speculative_tokens
        self.max_ngram_size = max_ngram_size
        self.device_group = device_group or [0]
        self.full_batch_size = full_batch_size
        self.prefill_bsz = prefill_bsz
        self.num_sessions = max(1, num_sessions)
        self.num_logits_to_keep = num_speculative_tokens + 1
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name, padding_side="right")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)
        
        # Initialize model session
        self._init_model()
        
        # Executor for parallel session use
        self._executor = ThreadPoolExecutor(max_workers=self.num_sessions)
    
    def _init_model(self):
        """Initialize the target model and session"""
        logger.info(f"Loading model: {self.target_model_name}")
        cache_path = Path(__file__).resolve().parent / ".qpc_cache.json"
        num_cores = max(1, 16 // self.num_sessions)
        cache_key = "|".join(
            [
                self.target_model_name,
                f"prefill={self.prefill_seq_len}",
                f"ctx={self.ctx_len}",
                f"spec={self.num_speculative_tokens}",
                f"full_bsz={self.full_batch_size}",
                f"num_devices={len(self.device_group)}",
                f"num_cores={num_cores}",
            ]
        )

        qpc_path = None
        if cache_path.is_file():
            try:
                cache_data = json.loads(cache_path.read_text())
                cached_path = cache_data.get(cache_key)
                if cached_path:
                    candidate = Path(cached_path)
                    if (candidate / "programqpc.bin").is_file():
                        qpc_path = candidate
                        logger.info(f"Using cached QPC at: {qpc_path}")
                    else:
                        cache_data.pop(cache_key, None)
                        cache_path.write_text(json.dumps(cache_data, indent=2))
            except (OSError, json.JSONDecodeError):
                pass

        if qpc_path is None:
            target_model = AutoModelForCausalLM.from_pretrained(
                self.target_model_name,
                continuous_batching=True,
                qaic_config={"speculative_model_type": "target"}
            )

            num_devices = len(self.device_group)
            qpc_path = target_model.compile(
                num_cores=num_cores,
                num_devices=num_devices,
                prefill_seq_len=self.prefill_seq_len,
                ctx_len=self.ctx_len,
                aic_enable_depth_first=True,
                full_batch_size=self.full_batch_size,
                num_speculative_tokens=self.num_speculative_tokens,
            )

            try:
                cache_data = {}
                if cache_path.is_file():
                    cache_data = json.loads(cache_path.read_text())
                cache_data[cache_key] = str(qpc_path)
                cache_path.write_text(json.dumps(cache_data, indent=2))
            except (OSError, json.JSONDecodeError):
                pass

        self.sessions = []
        self._session_buffers = {}
        self._session_queue: asyncio.Queue = asyncio.Queue()
        for _ in range(self.num_sessions):
            session = QAICInferenceSession(qpc_path, device_ids=self.device_group)

            # Skip KV cache buffers
            session.skip_buffers(
                set([x for x in session.input_names if x.startswith("past_")])
            )
            session.skip_buffers(
                set([x for x in session.output_names if x.endswith("_RetainedState")])
            )

            # Pre-allocate buffers per session
            prefill_buffer = np.zeros(
                (self.prefill_bsz, 1, self.vocab_size), dtype=np.float32
            )
            decode_buffer = np.zeros(
                (self.full_batch_size, self.num_logits_to_keep, self.vocab_size), dtype=np.float32
            )

            self.sessions.append(session)
            self._session_buffers[id(session)] = (prefill_buffer, decode_buffer)
            self._session_queue.put_nowait(session)
        
        logger.info("Model loaded successfully")
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a single prompt string"""
        if hasattr(self.tokenizer, "apply_chat_template"):
            template_messages = []
            for msg in messages:
                entry = {"role": msg.role, "content": msg.content}
                if msg.name:
                    entry["name"] = msg.name
                template_messages.append(entry)
            return self.tokenizer.apply_chat_template(
                template_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def _get_padded_input_len(self, input_len: int) -> int:
        num_chunks = -(input_len // -self.prefill_seq_len)
        input_len_padded = num_chunks * self.prefill_seq_len
        assert input_len_padded <= self.ctx_len, (
            f"Padded input length {input_len_padded} exceeds context length {self.ctx_len}"
        )
        return input_len_padded
    
    def _find_candidate_pred_tokens(
        self,
        input_ids: np.ndarray,
        fill_tok: int,
        max_ngram_size: int = 3,
        num_pred_tokens: int = 10
    ) -> tuple:
        decode_batch_size, input_length = input_ids.shape
        assert decode_batch_size == 1

        if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
            return np.full(num_pred_tokens, fill_tok, dtype=np.int64), True

        for ngram_size in range(max_ngram_size, 0, -1):
            ngram = input_ids[0, -ngram_size:]
            windows = np.lib.stride_tricks.sliding_window_view(
                input_ids[0], window_shape=ngram_size
            )
            matches = np.all(windows == ngram, axis=1)
            match_indices = np.where(matches)[0]

            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + num_pred_tokens
                if end_idx <= input_length and start_idx < input_length - ngram_size:
                    return input_ids[0, start_idx:end_idx], False

        return np.full(num_pred_tokens, fill_tok, dtype=np.int64), True
    
    def _run_prefill(self, session: QAICInferenceSession, inputs: dict, slot_idx: int) -> np.ndarray:
        input_len = inputs["input_ids"].shape[1]
        num_chunks = input_len // self.prefill_seq_len
        cache_index = np.array([[0]], np.int64)
        batch_index = np.array([[slot_idx]], np.int64)
        inputs["batch_index"] = batch_index

        for i in range(num_chunks):
            chunk_inputs = inputs.copy()
            chunk_inputs["input_ids"] = inputs["input_ids"][
                :, cache_index[0, 0]: cache_index[0, 0] + self.prefill_seq_len
            ]
            chunk_inputs["position_ids"] = inputs["position_ids"][
                :, cache_index[0, 0]: cache_index[0, 0] + self.prefill_seq_len
            ]
            outputs = session.run(chunk_inputs)
            cache_index += self.prefill_seq_len

        return outputs["logits"]
    
    def _run_inference_sync(
        self,
        session: QAICInferenceSession,
        prefill_buffer: np.ndarray,
        decode_buffer: np.ndarray,
        requests: List[InferenceRequest],
    ) -> List[Dict]:
        """Synchronous inference - runs in thread pool"""
        actual_batch_size = len(requests)
        decode_batch_size = self.full_batch_size
        
        logger.debug(f"Running inference for {actual_batch_size} requests")
        
        # Tokenize all prompts
        prompts_tokenized = []
        for req in requests:
            prompt = self._messages_to_prompt(req.messages)
            input_len = self.tokenizer(prompt, return_tensors="np", padding=True).input_ids.shape[1]
            input_len_padded = self._get_padded_input_len(input_len)
            p_tok = self.tokenizer(
                prompt, return_tensors="np", padding="max_length", max_length=input_len_padded
            )
            position_ids = np.where(
                p_tok.pop("attention_mask"),
                np.arange(input_len_padded),
                -1
            )
            p_tok["position_ids"] = position_ids
            p_tok["num_logits_to_keep"] = np.array([[1]], dtype=np.int64)
            prompts_tokenized.append(p_tok)
        
        # Pad with dummy requests if needed
        if actual_batch_size < decode_batch_size:
            dummy_tok = {k: v.copy() for k, v in prompts_tokenized[0].items()}
            for _ in range(decode_batch_size - actual_batch_size):
                prompts_tokenized.append({k: v.copy() for k, v in dummy_tok.items()})
        
        # Initialize tracking arrays
        generated_ids = [[] for _ in range(decode_batch_size)]
        input_lengths = [0] * decode_batch_size
        max_gen_lens = []
        for i in range(decode_batch_size):
            if i < actual_batch_size:
                max_gen_lens.append(requests[i].max_tokens)
            else:
                max_gen_lens.append(1)
        
        all_ids = np.zeros((decode_batch_size, self.ctx_len), dtype=np.int64)
        prompt_plus_gen_idx = np.zeros(decode_batch_size, dtype=np.int64)
        
        # Prepare decode inputs
        precode_inputs = {
            "input_ids": np.zeros((decode_batch_size, self.num_logits_to_keep), dtype=np.int64),
            "position_ids": np.zeros((decode_batch_size, self.num_logits_to_keep), dtype=np.int64),
            "batch_index": np.arange(decode_batch_size, dtype=np.int64).reshape(-1, 1),
            "num_logits_to_keep": np.full((self.num_logits_to_keep, 1), self.num_logits_to_keep - 1, dtype=np.int64),
        }
        
        # Run prefill for each slot
        session.set_buffers({"logits": prefill_buffer})
        for bi in range(decode_batch_size):
            logits = self._run_prefill(session, prompts_tokenized[bi], slot_idx=bi)
            input_ids = logits.argmax(2).astype(np.int64)
            generated_ids[bi].append(input_ids.item())
            precode_inputs["input_ids"][bi, 0] = input_ids.item()
            
            input_len = prompts_tokenized[bi]["position_ids"].max(1).item() + 1
            precode_inputs["position_ids"][bi] = np.arange(
                input_len, input_len + self.num_logits_to_keep, dtype=np.int64
            )
            input_lengths[bi] = input_len
            max_gen_lens[bi] = min(max_gen_lens[bi], self.ctx_len - input_len)
            
            all_ids[bi, :input_len + 1] = (
                prompts_tokenized[bi]["input_ids"][0, :input_len].tolist() + [input_ids.item()]
            )
            prompt_plus_gen_idx[bi] = input_len + 1
        
        # Decode phase
        session.set_buffers({"logits": decode_buffer})
        
        valid_batch_indices = np.zeros(decode_batch_size, dtype=bool)
        valid_batch_indices[:actual_batch_size] = True
        
        empty_indices = np.zeros(decode_batch_size, dtype=bool)
        position_ids_base = np.arange(self.num_logits_to_keep).reshape(1, -1).repeat(decode_batch_size, axis=0)
        
        iteration = 0
        while valid_batch_indices.any():
            iteration += 1
            
            # Generate n-gram proposals
            for bi in range(decode_batch_size):
                if not valid_batch_indices[bi]:
                    precode_inputs["position_ids"][bi, :] = -1
                    continue
                
                spec_tokens, has_empty_tokens = self._find_candidate_pred_tokens(
                    all_ids[bi:bi + 1, :prompt_plus_gen_idx[bi]],
                    fill_tok=-1,
                    max_ngram_size=self.max_ngram_size,
                    num_pred_tokens=self.num_speculative_tokens,
                )
                empty_indices[bi] = has_empty_tokens
                
                if has_empty_tokens:
                    precode_inputs["position_ids"][bi, 1:] = -1
                else:
                    precode_inputs["input_ids"][bi, 1:] = spec_tokens
            
            # Run target model
            outputs = session.run(precode_inputs)
            target_tokens = outputs["logits"].argmax(-1)
            
            # Verify predictions
            num_tokens_selected = np.ones(decode_batch_size, dtype=np.int64)
            precode_position_ids = np.full((decode_batch_size, self.num_logits_to_keep), -1, dtype=np.int64)
            non_empty_valid = ~empty_indices & valid_batch_indices
            
            if non_empty_valid.any():
                matching = (
                    precode_inputs["input_ids"][non_empty_valid, 1:] ==
                    target_tokens[non_empty_valid, :-1]
                )
                num_tokens_selected[non_empty_valid] = matching.cumprod(axis=1).sum(axis=1) + 1
            
            # Update position IDs
            for bi in range(decode_batch_size):
                if not valid_batch_indices[bi]:
                    continue
                if empty_indices[bi]:
                    precode_position_ids[bi] = position_ids_base[bi] + (
                        precode_inputs["position_ids"][bi, 0] + 1
                    )
                else:
                    precode_position_ids[bi] = precode_inputs["position_ids"][bi] + num_tokens_selected[bi]
            
            # Update generated tokens
            for bi in range(decode_batch_size):
                if not valid_batch_indices[bi]:
                    continue
                
                accepted = num_tokens_selected[bi]
                to_append = min(accepted, max_gen_lens[bi] - len(generated_ids[bi]))
                gen_ids = target_tokens[bi, :to_append]
                
                all_ids[bi, prompt_plus_gen_idx[bi]:prompt_plus_gen_idx[bi] + to_append] = gen_ids
                prompt_plus_gen_idx[bi] += to_append
                generated_ids[bi].extend(gen_ids.tolist())
                
                if (
                    len(generated_ids[bi]) >= max_gen_lens[bi] or
                    self.tokenizer.eos_token_id in gen_ids
                ):
                    valid_batch_indices[bi] = False
            
            if not valid_batch_indices.any():
                break
            
            # Update decode inputs
            for bi in range(decode_batch_size):
                if valid_batch_indices[bi]:
                    precode_inputs["input_ids"][bi, 0] = target_tokens[bi, num_tokens_selected[bi] - 1]
            
            precode_inputs["position_ids"] = precode_position_ids
        
        logger.debug(f"Inference completed in {iteration} iterations")
        
        # Build results only for actual requests
        results = []
        for bi in range(actual_batch_size):
            gen_ids = generated_ids[bi]
            if self.tokenizer.eos_token_id in gen_ids:
                eos_idx = gen_ids.index(self.tokenizer.eos_token_id)
                gen_ids = gen_ids[:eos_idx]
            
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append({
                "text": text,
                "token_ids": gen_ids,
                "prompt_tokens": input_lengths[bi],
                "completion_tokens": len(gen_ids),
            })
        
        return results
    
    async def run_batch(self, requests: List[InferenceRequest]) -> List[Dict]:
        """Run inference - dispatches to thread pool"""
        session = await self._session_queue.get()
        prefill_buffer, decode_buffer = self._session_buffers[id(session)]
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self._run_inference_sync,
                session,
                prefill_buffer,
                decode_buffer,
                requests,
            )
        finally:
            self._session_queue.put_nowait(session)


# =============================================================================
# FastAPI Application Factory
# =============================================================================


def create_app(
    target_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    scheduler_strategy: str = "fifo",
    max_batch_size: int = 8,
    prefill_seq_len: int = 256,
    ctx_len: int = 1024,
    num_speculative_tokens: int = 3,
    num_sessions: int = 1,
    device_group: List[int] = None,
) -> FastAPI:
    """Create and configure the FastAPI application"""
    
    # These will be initialized in lifespan
    state = {
        "scheduler": None,
        "engine": None,
        "model_name": target_model_name,
        "ctx_len": ctx_len,
        "num_speculative_tokens": num_speculative_tokens,
    }
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: Initialize engine and scheduler
        logger.info("Initializing inference engine...")
        
        engine = InferenceEngine(
            target_model_name=target_model_name,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            num_speculative_tokens=num_speculative_tokens,
            num_sessions=num_sessions,
            device_group=device_group or [0],
            full_batch_size=max_batch_size,
        )
        
        scheduler = RequestScheduler(
            strategy=scheduler_strategy,
            max_batch_size=max_batch_size,
        )
        scheduler.set_inference_engine(engine)
        
        state["engine"] = engine
        state["scheduler"] = scheduler
        
        await scheduler.start()
        
        logger.info("Server ready to accept requests")
        
        yield
        
        # Shutdown
        logger.info("Shutting down...")
        await scheduler.stop()
    
    app = FastAPI(
        title="Speculative Decoding API",
        description="OpenAI-compatible API with speculative decoding on Qualcomm Cloud AI 100",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ==========================================================================
    # API Endpoints
    # ==========================================================================
    
    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        return ModelList(
            data=[
                ModelInfo(
                    id=state["model_name"],
                    created=int(time.time()),
                    owned_by="qualcomm",
                )
            ]
        )
    
    @app.get("/v1/models/{model_id}", response_model=ModelInfo)
    async def get_model(model_id: str):
        if model_id != state["model_name"]:
            raise HTTPException(status_code=404, detail="Model not found")
        return ModelInfo(
            id=state["model_name"],
            created=int(time.time()),
            owned_by="qualcomm",
        )
    
    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        engine = state["engine"]
        scheduler = state["scheduler"]
        
        if engine is None or scheduler is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        
        # Validate model
        if request.model != state["model_name"] and request.model != "default":
            raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
        
        # Create prompt
        prompt = engine._messages_to_prompt(request.messages)
        prompt_tokens = len(engine.tokenizer.encode(prompt))
        
        # Calculate max tokens
        max_tokens = request.max_tokens or (state["ctx_len"] - prompt_tokens - 10)
        max_tokens = max(1, min(max_tokens, state["ctx_len"] - prompt_tokens - 1))
        
        # Create inference request
        inference_request = InferenceRequest(
            request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            prompt=prompt,
            messages=request.messages,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            num_speculative_tokens=request.num_speculative_tokens or state["num_speculative_tokens"],
            priority=request.priority or 0,
            created_at=time.time(),
            is_streaming=request.stream,
        )
        
        if request.stream:
            # Streaming: submit to scheduler, then stream tokens from queue
            async def generate_stream():
                try:
                    # Submit request to scheduler (non-blocking)
                    await scheduler.submit_request(inference_request)
                    
                    # Wait for completion and get result
                    result = await scheduler.wait_for_result(inference_request)
                    
                    # Stream tokens one by one
                    token_ids = result.get("token_ids", [])
                    for token_id in token_ids:
                        text = engine.tokenizer.decode([token_id], skip_special_tokens=True)
                        if text:
                            chunk = ChatCompletionChunk(
                                id=inference_request.request_id,
                                created=int(time.time()),
                                model=request.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta={"content": text},
                                        finish_reason=None,
                                    )
                                ],
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"
                        # Small delay for streaming effect
                        await asyncio.sleep(0.005)
                    
                    # Final chunk
                    final_chunk = ChatCompletionChunk(
                        id=inference_request.request_id,
                        created=int(time.time()),
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta={},
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    logger.exception(f"Streaming error: {e}")
                    yield f"data: {{'error': '{str(e)}'}}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        
        else:
            # Non-streaming: submit and wait
            await scheduler.submit_request(inference_request)
            result = await scheduler.wait_for_result(inference_request)
            
            return ChatCompletionResponse(
                id=inference_request.request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=result["text"],
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    total_tokens=result["prompt_tokens"] + result["completion_tokens"],
                ),
            )
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "model": state["model_name"],
            "ready": state["engine"] is not None,
        }
    
    @app.get("/v1/stats")
    async def get_stats():
        if state["scheduler"] is None:
            return {"error": "Scheduler not initialized"}
        return state["scheduler"].get_stats()
    
    @app.get("/")
    async def root():
        return {
            "message": "Speculative Decoding API Server",
            "docs": "/docs",
            "openapi": "/openapi.json",
        }
    
    return app


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI-Compatible API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Target model name",
    )
    parser.add_argument(
        "--scheduler-strategy",
        type=str,
        choices=["fifo", "priority", "dynamic"],
        default="fifo",
        help="Scheduling strategy",
    )
    parser.add_argument("--max-batch-size", type=int, default=8, help="Maximum batch size")
    parser.add_argument("--prefill-seq-len", type=int, default=256, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=1024, help="Context length")
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=3,
        help="Number of speculative tokens",
    )
    parser.add_argument(
        "--num-sessions",
        type=int,
        default=1,
        help="Number of QAIC inference sessions to create",
    )
    parser.add_argument(
        "--device-group",
        type=str,
        default="0",
        help="Comma-separated device IDs",
    )
    
    args = parser.parse_args()
    
    device_group = [int(d) for d in args.device_group.split(",")]
    
    app = create_app(
        target_model_name=args.model,
        scheduler_strategy=args.scheduler_strategy,
        max_batch_size=args.max_batch_size,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_speculative_tokens=args.num_speculative_tokens,
        num_sessions=args.num_sessions,
        device_group=device_group,
    )
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
