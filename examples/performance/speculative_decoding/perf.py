#!/usr/bin/env python3
"""
OpenAI API Performance Test Script
Tests throughput, latency, and tokens per second
"""

import os
import time
import json
import asyncio
import aiohttp
import argparse
import statistics
from dataclasses import dataclass, field
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("MODEL", "gpt-3.5-turbo")


@dataclass
class RequestResult:
    """Store results from a single request"""
    success: bool
    latency: float  # seconds
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token: float = 0.0  # for streaming
    error: Optional[str] = None


@dataclass
class PerformanceReport:
    """Aggregated performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0
    latencies: List[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    errors: List[str] = field(default_factory=list)


# Test prompts with varying complexity
TEST_PROMPTS = [
    {"prompt": "Say 'Hi'", "expected_tokens": 5},
    {"prompt": "What is 2+2? Answer in one word.", "expected_tokens": 10},
    {"prompt": "List 3 colors.", "expected_tokens": 20},
    {"prompt": "Explain Python in 2 sentences.", "expected_tokens": 50},
    {"prompt": "Write a haiku about coding.", "expected_tokens": 30},
    {"prompt": "What is machine learning? Brief answer.", "expected_tokens": 50},
    {"prompt": "Count from 1 to 10.", "expected_tokens": 30},
    {"prompt": "Name 5 programming languages.", "expected_tokens": 25},
]


def create_payload(prompt: str, stream: bool = False) -> dict:
    """Create the API request payload"""
    return {
        "model": MODEL,
        "stream": stream,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }


def get_headers() -> dict:
    """Get API request headers"""
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }


# ============== Synchronous Testing ==============

def sync_request(prompt: str) -> RequestResult:
    """Make a synchronous API request"""
    url = f"{OPENAI_BASE_URL}/v1/chat/completions"
    payload = create_payload(prompt, stream=False)
    
    start_time = time.perf_counter()
    
    try:
        response = requests.post(
            url,
            headers=get_headers(),
            json=payload,
            timeout=60
        )
        
        latency = time.perf_counter() - start_time
        
        if response.status_code == 200:
            data = response.json()
            usage = data.get("usage", {})
            return RequestResult(
                success=True,
                latency=latency,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0)
            )
        else:
            return RequestResult(
                success=False,
                latency=latency,
                error=f"HTTP {response.status_code}: {response.text[:200]}"
            )
    except Exception as e:
        latency = time.perf_counter() - start_time
        return RequestResult(
            success=False,
            latency=latency,
            error=str(e)
        )


def run_sync_test(num_requests: int, prompts: List[str]) -> PerformanceReport:
    """Run synchronous sequential test"""
    print(f"\n{'='*60}")
    print("Running SYNCHRONOUS Sequential Test")
    print(f"{'='*60}")
    
    report = PerformanceReport()
    start_time = time.perf_counter()
    
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        print(f"  Request {i+1}/{num_requests}...", end=" ", flush=True)
        
        result = sync_request(prompt)
        
        report.total_requests += 1
        if result.success:
            report.successful_requests += 1
            report.latencies.append(result.latency)
            report.prompt_tokens += result.prompt_tokens
            report.completion_tokens += result.completion_tokens
            report.total_tokens += result.total_tokens
            print(f"✓ {result.latency:.2f}s ({result.completion_tokens} tokens)")
        else:
            report.failed_requests += 1
            report.errors.append(result.error)
            print(f"✗ {result.error[:50]}")
    
    report.total_time = time.perf_counter() - start_time
    return report


# ============== Concurrent Testing with ThreadPool ==============

def run_concurrent_test(num_requests: int, concurrency: int, prompts: List[str]) -> PerformanceReport:
    """Run concurrent test using ThreadPoolExecutor"""
    print(f"\n{'='*60}")
    print(f"Running CONCURRENT Test (concurrency={concurrency})")
    print(f"{'='*60}")
    
    report = PerformanceReport()
    request_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]
    
    start_time = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(sync_request, request_prompts))
    
    report.total_time = time.perf_counter() - start_time
    
    for result in results:
        report.total_requests += 1
        if result.success:
            report.successful_requests += 1
            report.latencies.append(result.latency)
            report.prompt_tokens += result.prompt_tokens
            report.completion_tokens += result.completion_tokens
            report.total_tokens += result.total_tokens
        else:
            report.failed_requests += 1
            report.errors.append(result.error)
    
    return report


# ============== Async Testing ==============

async def async_request(session: aiohttp.ClientSession, prompt: str) -> RequestResult:
    """Make an async API request"""
    url = f"{OPENAI_BASE_URL}/v1/chat/completions"
    payload = create_payload(prompt, stream=False)
    
    start_time = time.perf_counter()
    
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
            latency = time.perf_counter() - start_time
            
            if response.status == 200:
                data = await response.json()
                usage = data.get("usage", {})
                return RequestResult(
                    success=True,
                    latency=latency,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0)
                )
            else:
                text = await response.text()
                return RequestResult(
                    success=False,
                    latency=latency,
                    error=f"HTTP {response.status}: {text[:200]}"
                )
    except Exception as e:
        latency = time.perf_counter() - start_time
        return RequestResult(
            success=False,
            latency=latency,
            error=str(e)
        )


async def run_async_test(num_requests: int, concurrency: int, prompts: List[str]) -> PerformanceReport:
    """Run async concurrent test"""
    print(f"\n{'='*60}")
    print(f"Running ASYNC Test (concurrency={concurrency})")
    print(f"{'='*60}")
    
    report = PerformanceReport()
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(session: aiohttp.ClientSession, prompt: str) -> RequestResult:
        async with semaphore:
            return await async_request(session, prompt)
    
    start_time = time.perf_counter()
    
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(headers=get_headers(), connector=connector) as session:
        tasks = [
            bounded_request(session, prompts[i % len(prompts)])
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    report.total_time = time.perf_counter() - start_time
    
    for result in results:
        report.total_requests += 1
        if isinstance(result, Exception):
            report.failed_requests += 1
            report.errors.append(str(result))
        elif result.success:
            report.successful_requests += 1
            report.latencies.append(result.latency)
            report.prompt_tokens += result.prompt_tokens
            report.completion_tokens += result.completion_tokens
            report.total_tokens += result.total_tokens
        else:
            report.failed_requests += 1
            report.errors.append(result.error)
    
    return report


# ============== Streaming Test ==============

async def async_streaming_request(session: aiohttp.ClientSession, prompt: str) -> RequestResult:
    """Make an async streaming API request"""
    url = f"{OPENAI_BASE_URL}/v1/chat/completions"
    payload = create_payload(prompt, stream=True)
    
    start_time = time.perf_counter()
    time_to_first_token = 0.0
    token_count = 0
    
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status == 200:
                first_token_received = False
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        if not first_token_received:
                            time_to_first_token = time.perf_counter() - start_time
                            first_token_received = True
                        try:
                            data = json.loads(line[6:])
                            if data.get('choices', [{}])[0].get('delta', {}).get('content'):
                                token_count += 1
                        except json.JSONDecodeError:
                            pass
                
                latency = time.perf_counter() - start_time
                return RequestResult(
                    success=True,
                    latency=latency,
                    completion_tokens=token_count,
                    time_to_first_token=time_to_first_token
                )
            else:
                text = await response.text()
                return RequestResult(
                    success=False,
                    latency=time.perf_counter() - start_time,
                    error=f"HTTP {response.status}: {text[:200]}"
                )
    except Exception as e:
        return RequestResult(
            success=False,
            latency=time.perf_counter() - start_time,
            error=str(e)
        )


async def run_streaming_test(num_requests: int, concurrency: int, prompts: List[str]) -> PerformanceReport:
    """Run streaming test"""
    print(f"\n{'='*60}")
    print(f"Running STREAMING Test (concurrency={concurrency})")
    print(f"{'='*60}")
    
    report = PerformanceReport()
    ttft_list = []
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(session: aiohttp.ClientSession, prompt: str) -> RequestResult:
        async with semaphore:
            return await async_streaming_request(session, prompt)
    
    start_time = time.perf_counter()
    
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(headers=get_headers(), connector=connector) as session:
        tasks = [
            bounded_request(session, prompts[i % len(prompts)])
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    report.total_time = time.perf_counter() - start_time
    
    for result in results:
        report.total_requests += 1
        if isinstance(result, Exception):
            report.failed_requests += 1
            report.errors.append(str(result))
        elif result.success:
            report.successful_requests += 1
            report.latencies.append(result.latency)
            report.completion_tokens += result.completion_tokens
            if result.time_to_first_token > 0:
                ttft_list.append(result.time_to_first_token)
        else:
            report.failed_requests += 1
            report.errors.append(result.error)
    
    # Print TTFT metrics
    if ttft_list:
        print(f"\n  Time to First Token (TTFT):")
        print(f"    Mean: {statistics.mean(ttft_list)*1000:.2f} ms")
        print(f"    Median: {statistics.median(ttft_list)*1000:.2f} ms")
        if len(ttft_list) > 1:
            print(f"    Std Dev: {statistics.stdev(ttft_list)*1000:.2f} ms")
    
    return report


# ============== Reporting ==============

def print_report(report: PerformanceReport, test_name: str = ""):
    """Print a formatted performance report"""
    print(f"\n{'='*60}")
    print(f"PERFORMANCE REPORT {test_name}")
    print(f"{'='*60}")
    
    print(f"\n📊 Request Statistics:")
    print(f"  Total Requests:      {report.total_requests}")
    print(f"  Successful:          {report.successful_requests}")
    print(f"  Failed:              {report.failed_requests}")
    success_rate = (report.successful_requests / report.total_requests * 100) if report.total_requests > 0 else 0
    print(f"  Success Rate:        {success_rate:.1f}%")
    
    print(f"\n⏱️  Timing Metrics:")
    print(f"  Total Time:          {report.total_time:.2f} seconds")
    
    if report.latencies:
        print(f"  Throughput:          {report.successful_requests / report.total_time:.2f} req/s")
        print(f"\n  Latency (per request):")
        print(f"    Mean:              {statistics.mean(report.latencies)*1000:.2f} ms")
        print(f"    Median:            {statistics.median(report.latencies)*1000:.2f} ms")
        print(f"    Min:               {min(report.latencies)*1000:.2f} ms")
        print(f"    Max:               {max(report.latencies)*1000:.2f} ms")
        if len(report.latencies) > 1:
            print(f"    Std Dev:           {statistics.stdev(report.latencies)*1000:.2f} ms")
            # Calculate percentiles
            sorted_latencies = sorted(report.latencies)
            p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
            p90 = sorted_latencies[int(len(sorted_latencies) * 0.90)]
            p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99 = sorted_latencies[min(int(len(sorted_latencies) * 0.99), len(sorted_latencies)-1)]
            print(f"    P50:               {p50*1000:.2f} ms")
            print(f"    P90:               {p90*1000:.2f} ms")
            print(f"    P95:               {p95*1000:.2f} ms")
            print(f"    P99:               {p99*1000:.2f} ms")
    
    print(f"\n🔤 Token Statistics:")
    print(f"  Total Tokens:        {report.total_tokens}")
    print(f"  Prompt Tokens:       {report.prompt_tokens}")
    print(f"  Completion Tokens:   {report.completion_tokens}")
    
    if report.total_time > 0:
        print(f"  Tokens/second:       {report.total_tokens / report.total_time:.2f}")
        print(f"  Completion tok/s:    {report.completion_tokens / report.total_time:.2f}")
    
    if report.errors:
        print(f"\n❌ Errors ({len(report.errors)} total):")
        unique_errors = list(set(report.errors))[:5]  # Show first 5 unique errors
        for error in unique_errors:
            print(f"  - {error[:80]}...")


def warmup(num_requests: int = 3):
    """Warm up the API connection"""
    print("\n🔥 Warming up API connection...")
    for i in range(num_requests):
        result = sync_request("Hi")
        status = "✓" if result.success else "✗"
        print(f"  Warmup {i+1}/{num_requests}: {status}")
    print("  Warmup complete!\n")


def main():
    parser = argparse.ArgumentParser(description="OpenAI API Performance Test")
    parser.add_argument("-n", "--num-requests", type=int, default=10,
                        help="Number of requests to send (default: 10)")
    parser.add_argument("-c", "--concurrency", type=int, default=5,
                        help="Number of concurrent requests (default: 5)")
    parser.add_argument("--sync", action="store_true",
                        help="Run synchronous sequential test")
    parser.add_argument("--concurrent", action="store_true",
                        help="Run concurrent test with ThreadPool")
    parser.add_argument("--async", dest="run_async", action="store_true",
                        help="Run async concurrent test")
    parser.add_argument("--stream", action="store_true",
                        help="Run streaming test")
    parser.add_argument("--all", action="store_true",
                        help="Run all test types")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup requests")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Override OPENAI_BASE_URL")
    parser.add_argument("--model", type=str, default=None,
                        help="Override MODEL")
    
    args = parser.parse_args()
    
    # Override environment variables if provided
    global OPENAI_BASE_URL, MODEL
    if args.base_url:
        OPENAI_BASE_URL = args.base_url
    if args.model:
        MODEL = args.model
    
    # Validate configuration
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable is not set")
        return
    
    print("="*60)
    print("OpenAI API Performance Test")
    print("="*60)
    print(f"\n📋 Configuration:")
    print(f"  Base URL:     {OPENAI_BASE_URL}")
    print(f"  Model:        {MODEL}")
    print(f"  Requests:     {args.num_requests}")
    print(f"  Concurrency:  {args.concurrency}")
    
    # Extract prompts
    prompts = [p["prompt"] for p in TEST_PROMPTS]
    
    # Warmup
    if not args.no_warmup:
        warmup()
    
    # Default to async test if no specific test selected
    run_all = args.all or not (args.sync or args.concurrent or args.run_async or args.stream)
    
    # Run tests
    if args.sync or run_all:
        report = run_sync_test(args.num_requests, prompts)
        print_report(report, "[Sequential]")
    
    if args.concurrent or run_all:
        report = run_concurrent_test(args.num_requests, args.concurrency, prompts)
        print_report(report, "[Concurrent ThreadPool]")
    
    if args.run_async or run_all:
        report = asyncio.run(run_async_test(args.num_requests, args.concurrency, prompts))
        print_report(report, "[Async]")
    
    if args.stream or run_all:
        report = asyncio.run(run_streaming_test(args.num_requests, args.concurrency, prompts))
        print_report(report, "[Streaming]")
    
    print(f"\n{'='*60}")
    print("Test Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
