# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
OpenAI-compatible API server using non-continuous batching streaming.
"""

import asyncio
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.text_generation_inference import TextGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class RequestStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    messages: List[ChatMessage]
    prompt_tokens: int
    max_tokens: int
    temperature: float
    top_p: float
    created_at: float
    is_streaming: bool = False
    status: RequestStatus = RequestStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completion_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


class NonCBStreamingEngine:
    def __init__(
        self,
        model_name: str,
        prefill_seq_len: int,
        ctx_len: int,
        generation_len: int,
        num_cores: int,
        device_group: Optional[List[int]],
    ):
        self.model_name = model_name
        self.ctx_len = ctx_len
        self.default_generation_len = generation_len
        self.device_group = device_group
        self._lock = threading.Lock()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = QEFFAutoModelForCausalLM.from_pretrained(model_name, continuous_batching=False)
        self.qpc_path = self.model.compile(
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            num_cores=num_cores,
            num_devices=(1 if device_group is None else len(device_group)),
        )
        self._text_generation = TextGeneration(
            tokenizer=self.tokenizer,
            qpc_path=self.qpc_path,
            ctx_len=ctx_len,
            device_id=device_group,
        )

    def stream_tokens(self, prompt: str, max_tokens: Optional[int]) -> Generator[str, None, None]:
        generation_len = max_tokens or self.default_generation_len
        with self._lock:
            for token_list in self._text_generation.generate_stream_tokens([prompt], generation_len):
                if not token_list:
                    continue
                token = token_list[0]
                if token:
                    yield token


def messages_to_prompt(tokenizer, messages: List[ChatMessage]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        template_messages = []
        for msg in messages:
            entry = {"role": msg.role, "content": msg.content}
            if msg.name:
                entry["name"] = msg.name
            template_messages.append(entry)
        return tokenizer.apply_chat_template(template_messages, tokenize=False, add_generation_prompt=True)

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


def create_app(
    target_model_name: str = "Qwen/Qwen2-1.5B-Instruct",
    prefill_seq_len: int = 128,
    ctx_len: int = 512,
    generation_len: int = 128,
    num_cores: int = 16,
    device_group: Optional[List[int]] = None,
) -> FastAPI:
    state = {
        "stream_engine": None,
        "model_name": target_model_name,
        "ctx_len": ctx_len,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Initializing non-continuous batching streaming engine...")
        stream_engine = NonCBStreamingEngine(
            model_name=target_model_name,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            generation_len=generation_len,
            num_cores=num_cores,
            device_group=device_group or [0],
        )
        state["stream_engine"] = stream_engine
        logger.info("Server ready to accept requests")
        yield
        logger.info("Shutting down...")

    app = FastAPI(
        title="Continuous Batching API",
        description="OpenAI-compatible API for continuous batching inference",
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

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        stream_engine = state["stream_engine"]
        if stream_engine is None:
            raise HTTPException(status_code=503, detail="Server not ready")

        if request.model != state["model_name"] and request.model != "default":
            raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

        prompt = messages_to_prompt(stream_engine.tokenizer, request.messages)
        prompt_tokens = len(stream_engine.tokenizer.encode(prompt))
        max_tokens = request.max_tokens or (state["ctx_len"] - prompt_tokens - 10)
        max_tokens = max(1, min(max_tokens, state["ctx_len"] - prompt_tokens - 1))

        inference_request = InferenceRequest(
            request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            prompt=prompt,
            messages=request.messages,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            created_at=time.time(),
            is_streaming=request.stream,
        )

        if request.stream:
            async def generate_stream():
                queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_event_loop()

                def run_stream():
                    try:
                        for token in stream_engine.stream_tokens(prompt, max_tokens):
                            loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                    except Exception as exc:
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

                threading.Thread(target=run_stream, daemon=True).start()

                while True:
                    kind, payload = await queue.get()
                    if kind == "token":
                        chunk = ChatCompletionChunk(
                            id=inference_request.request_id,
                            created=int(time.time()),
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta={"content": payload},
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    elif kind == "done":
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
                        break
                    else:
                        yield f"data: {{\"error\": \"{payload}\"}}\n\n"
                        break

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        completion_parts: List[str] = []
        for token in stream_engine.stream_tokens(prompt, max_tokens):
            completion_parts.append(token)
        completion_text = "".join(completion_parts)
        completion_tokens = len(stream_engine.tokenizer.encode(completion_text))

        return ChatCompletionResponse(
            id=inference_request.request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=completion_text,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "model": state["model_name"],
            "ready": state["stream_engine"] is not None,
        }

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI-Compatible API Server (non-CB streaming)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument("--prefill-seq-len", type=int, default=128, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=512, help="Context length")
    parser.add_argument("--generation-len", type=int, default=128, help="Default generation length")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument(
        "--device-group",
        type=str,
        default="0",
        help="Comma-separated device IDs",
    )

    args = parser.parse_args()
    device_group = [int(d) for d in args.device_group.split(",")]

    app = create_app(
        target_model_name=args.model_name,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        generation_len=args.generation_len,
        num_cores=args.num_cores,
        device_group=device_group,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
