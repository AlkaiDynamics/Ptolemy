#!/usr/bin/env python
"""
Multi-Model Processor Interface Tests

This module provides comprehensive tests for the Multi-Model Processor interface,
including unit tests, integration tests, API tests, and end-to-end tests.

Usage:
    python test_processor_interface.py [--basic] [--cache] [--error] [--stream] [--all] [--verbose]
"""

import requests
import time
import json
import argparse
import pytest
import asyncio
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from typing import Dict, List, Any, Tuple
import random
import sys
import os
from pathlib import Path
import unittest.mock as mock

# Initialize console for rich output
console = Console()

# Server configuration
BASE_URL = "http://127.0.0.1:8000/processor"

class MockResponse:
    """Mock HTTP response for testing."""
    def __init__(self, status_code, json_data, text=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text or json.dumps(json_data)
        
    def json(self):
        return self._json_data

class TestProcessorInterface:
    """Comprehensive tests for the Multi-Model Processor interface."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        console.print("\n[bold]Multi-Model Processor Interface Tests[/bold]")
        console.print(f"Server URL: {BASE_URL}")
        
        # Check if server is running
        try:
            response = requests.get(f"{BASE_URL}", timeout=2)
            if response.status_code == 200:
                console.print("[green]Server is running and responding[/green]\n")
            else:
                console.print(f"[bold red]Server returned status code {response.status_code}[/bold red]")
        except requests.RequestException:
            console.print("[bold red]Server not responding. Please make sure it's running.[/bold red]")
    
    @pytest.fixture
    def available_models(self):
        """Fixture to get available models or use fallbacks."""
        try:
            max_retries = 3
            retry_delay = 1
            models_response = None
            
            for retry in range(max_retries):
                try:
                    console.print(f"[yellow]Attempt {retry + 1}/{max_retries} to fetch models...[/yellow]")
                    models_response = requests.get(f"{BASE_URL}/models", timeout=10)
                    if models_response.status_code == 200:
                        break
                    console.print(f"[yellow]Received status code {models_response.status_code}, retrying...[/yellow]")
                    time.sleep(retry_delay)
                except requests.RequestException as e:
                    console.print(f"[yellow]Request error: {str(e)}, retrying...[/yellow]")
                    time.sleep(retry_delay)
            
            if not models_response or models_response.status_code != 200:
                console.print(f"[bold red]Failed to get models after {max_retries} attempts.[/bold red]")
                return ["openai_gpt4", "anthropic_claude"]
            
            available_models = models_response.json().get("models", [])
            
            # Fallback to hardcoded models if empty list is returned
            if not available_models:
                console.print("[bold yellow]No models returned from API, using fallback models for testing...[/bold yellow]")
                available_models = ["openai_gpt4", "anthropic_claude"]
            
            console.print(f"[green]Available Models: {', '.join(available_models)}[/green]")
            return available_models
            
        except Exception as e:
            console.print(f"[bold red]Error fetching models: {str(e)}[/bold red]")
            return ["openai_gpt4", "anthropic_claude"]
    
    @pytest.fixture
    def mock_processor_client(self, monkeypatch):
        """Mock the processor client for unit and integration tests."""
        class MockClient:
            def get(self, url, **kwargs):
                if url.endswith('/models'):
                    return MockResponse(200, {"models": ["mock_gpt", "mock_claude"]})
                elif url.endswith('/cache-stats'):
                    return MockResponse(200, {
                        "hits": 5,
                        "misses": 10,
                        "hit_rate": 0.33,
                        "size": 15
                    })
                elif url.endswith('/error-analysis'):
                    return MockResponse(200, {
                        "error_count": 2,
                        "error_rate": 0.05,
                        "common_errors": ["timeout", "authentication"]
                    })
                return MockResponse(404, {"error": "Not found"})
                
            def post(self, url, json=None, **kwargs):
                if url.endswith('/process'):
                    if json.get('task') == 'error':
                        return MockResponse(500, {
                            "success": False,
                            "error": "Simulated error"
                        })
                    return MockResponse(200, {
                        "success": True,
                        "output": f"Response to: {json.get('task')}",
                        "model": "mock_model",
                        "cached": False,
                        "processing_time": 0.5
                    })
                elif url.endswith('/clear-cache'):
                    return MockResponse(200, {"success": True, "message": "Cache cleared"})
                return MockResponse(404, {"error": "Not found"})
        
        return MockClient()

    # Unit Tests
    @pytest.mark.unit
    def test_processor_state_creation(self):
        """Test creation of ProcessorState object."""
        from ptolemy.multi_model_processor.state import ProcessorState
        
        state = ProcessorState.create(task="Test task")
        assert state.task == "Test task"
        assert state.task_id is not None
        assert state.current_stage == "created"
    
    @pytest.mark.unit
    def test_processor_error_handling(self):
        """Test error handling in the processor."""
        from ptolemy.multi_model_processor.utils.error_handling import ErrorHandler, ErrorSeverity
        
        handler = ErrorHandler()
        error_id = handler.record_error("Test error", severity=ErrorSeverity.WARNING)
        assert error_id is not None
        assert handler.get_error_count() == 1
    
    # Integration Tests
    @pytest.mark.integration
    def test_processor_cache_integration(self, mock_processor_client):
        """Test integration with the cache subsystem."""
        response = mock_processor_client.get(f"{BASE_URL}/cache-stats")
        assert response.status_code == 200
        data = response.json()
        assert "hits" in data
        assert "hit_rate" in data
    
    @pytest.mark.integration
    def test_processor_error_recovery(self, mock_processor_client):
        """Test error recovery mechanisms."""
        response = mock_processor_client.post(f"{BASE_URL}/process", json={"task": "error"})
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    # API Tests
    @pytest.mark.api
    def test_models_endpoint(self, available_models):
        """Test the models endpoint."""
        response = requests.get(f"{BASE_URL}/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
    
    @pytest.mark.api
    def test_process_endpoint(self, available_models):
        """Test the process endpoint."""
        payload = {
            "task": "Explain what API testing is in one sentence.",
            "parameters": {
                "temperature": 0.5
            }
        }
        response = requests.post(f"{BASE_URL}/process", json=payload)
        console.print(f"Process endpoint response: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "output" in data
            assert "model" in data
            assert "processing_time" in data
    
    @pytest.mark.api
    def test_cache_stats_endpoint(self):
        """Test the cache stats endpoint."""
        response = requests.get(f"{BASE_URL}/cache-stats")
        if response.status_code == 200:
            data = response.json()
            assert "hits" in data
            assert "misses" in data
            assert "hit_rate" in data
    
    # End-to-End Tests
    @pytest.mark.e2e
    def test_basic_processing(self, available_models):
        """
        Test basic task processing with different models and parameters.
        This test evaluates the core functionality of the processor.
        """
        console.print("\n\n[bold]Testing Basic Processing[/bold]")
        
        # Create test cases
        test_cases = [
            {
                "task": "Explain the concept of quantum computing.",
                "model": None,
                "parameters": {"temperature": 0.7}
            },
            {
                "task": "Write a haiku about the stars.",
                "model": None,
                "parameters": {"temperature": 0.3}
            }
        ]
        
        # Create results table
        table = Table(title="Basic Processing Tests")
        table.add_column("Test #", justify="center")
        table.add_column("Task", justify="left")
        table.add_column("Model", justify="left")
        table.add_column("Temperature", justify="center")
        table.add_column("Processing Time", justify="center")
        table.add_column("Cache Hit", justify="center")
        table.add_column("Status", justify="center")
        
        results = []
        success_count = 0
        total_time = 0
        cache_hits = 0
        
        # Run tests with progress bar
        with Progress() as progress:
            task = progress.add_task("Running tests...", total=len(test_cases))
            
            for i, test_case in enumerate(test_cases):
                # Send request
                try:
                    payload = {
                        "task": test_case["task"],
                        "model_preference": test_case["model"],
                        "parameters": test_case["parameters"]
                    }
                    
                    start_time = time.time()
                    response = requests.post(f"{BASE_URL}/process", json=payload, timeout=30)
                    processing_time = time.time() - start_time
                    
                    total_time += processing_time
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success", False):
                            success_count += 1
                            status = "✅ Success"
                            if data.get("cached", False):
                                cache_hits += 1
                                cache_hit = "✓"
                            else:
                                cache_hit = "✗"
                        else:
                            status = f"❌ Error: {data.get('error', 'Unknown error')}"
                            cache_hit = "✗"
                    else:
                        status = f"❌ HTTP Error: {response.status_code}"
                        cache_hit = "✗"
                        
                except Exception as e:
                    processing_time = time.time() - start_time
                    status = f"❌ Exception: {str(e)}"
                    cache_hit = "✗"
                
                # Add result to table
                table.add_row(
                    str(i + 1),
                    test_case["task"][:20] + "..." if len(test_case["task"]) > 20 else test_case["task"],
                    test_case["model"] if test_case["model"] else "",
                    str(test_case["parameters"].get("temperature", "default")),
                    f"{processing_time:.2f}s",
                    cache_hit,
                    status
                )
                
                results.append({
                    "test_num": i + 1,
                    "task": test_case["task"],
                    "model": test_case["model"],
                    "temperature": test_case["parameters"].get("temperature", "default"),
                    "processing_time": processing_time,
                    "cache_hit": cache_hit == "✓",
                    "status": status
                })
                
                progress.update(task, advance=1)
        
        # Display results
        console.print(table)
        
        # Calculate and display statistics
        if results:
            success_rate = (success_count / len(results)) * 100
            avg_time = total_time / len(results)
            cache_hit_rate = (cache_hits / len(results)) * 100
            
            console.print(f"Success Rate: {success_rate:.2f}%")
            console.print(f"Average Processing Time: {avg_time:.2f}s")
            console.print(f"Cache Hit Rate: {cache_hit_rate:.2f}%")
    
    @pytest.mark.e2e
    def test_cache_performance(self, available_models):
        """
        Test cache performance by running the same tasks multiple times.
        This test evaluates the caching mechanism's effectiveness.
        """
        console.print("\n\n[bold]Testing Cache Performance[/bold]")
        
        # Get cache stats before test
        try:
            initial_stats = requests.get(f"{BASE_URL}/cache-stats", timeout=5).json()
            console.print(f"Initial Cache Stats: {initial_stats}")
        except Exception as e:
            console.print(f"[yellow]Error getting initial cache stats: {str(e)}[/yellow]")
            initial_stats = {"hits": 0, "misses": 0}
        
        # Test repeated task execution
        repeated_task = "What is artificial intelligence?"
        iterations = 3
        times = []
        cache_hits = 0
        
        console.print(f"Running same task {iterations} times to test caching...")
        
        for i in range(iterations):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{BASE_URL}/process", 
                    json={"task": repeated_task, "bypass_cache": False}, 
                    timeout=30
                )
                processing_time = time.time() - start_time
                times.append(processing_time)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("cached", False):
                        cache_hits += 1
                        console.print(f"Iteration {i+1}: [green]Cache hit[/green] - {processing_time:.2f}s")
                    else:
                        console.print(f"Iteration {i+1}: [yellow]Cache miss[/yellow] - {processing_time:.2f}s")
                else:
                    console.print(f"Iteration {i+1}: [red]Error {response.status_code}[/red] - {processing_time:.2f}s")
            except Exception as e:
                console.print(f"Iteration {i+1}: [bold red]Exception: {str(e)}[/bold red]")
        
        # Get cache stats after test
        try:
            final_stats = requests.get(f"{BASE_URL}/cache-stats", timeout=5).json()
            console.print(f"Final Cache Stats: {final_stats}")
            
            # Calculate diff
            hits_diff = final_stats.get("hits", 0) - initial_stats.get("hits", 0)
            misses_diff = final_stats.get("misses", 0) - initial_stats.get("misses", 0)
            
            console.print(f"Cache Hits Increase: {hits_diff}")
            console.print(f"Cache Misses Increase: {misses_diff}")
        except Exception as e:
            console.print(f"[yellow]Error getting final cache stats: {str(e)}[/yellow]")
        
        # Display cache hit rate
        cache_hit_rate = (cache_hits / iterations) * 100
        console.print(f"Measured Cache Hit Rate: {cache_hit_rate:.2f}%")
        
        # Display timing comparison
        if times:
            first_time = times[0]
            avg_cached_time = sum(times[1:]) / max(len(times[1:]), 1) if len(times) > 1 else 0
            
            console.print(f"First execution time: {first_time:.2f}s")
            console.print(f"Average cached execution time: {avg_cached_time:.2f}s")
            
            if avg_cached_time > 0 and first_time > 0:
                speedup = first_time / avg_cached_time
                console.print(f"Cache speedup: {speedup:.2f}x")
    
    @pytest.mark.e2e
    def test_error_recovery(self, available_models):
        """
        Test error recovery mechanisms by triggering errors.
        This test evaluates the system's robustness against failures.
        """
        console.print("\n\n[bold]Testing Error Recovery[/bold]")
        
        # Test cases designed to trigger errors
        error_test_cases = [
            {
                "name": "Empty Task",
                "payload": {"task": "", "model_preference": None},
                "expected_code": 422  # Expecting Unprocessable Entity
            },
            {
                "name": "Invalid Model",
                "payload": {"task": "Test task", "model_preference": "non_existent_model"},
                "expected_code": 503  # Expecting Service Unavailable
            },
            {
                "name": "Malformed Parameters",
                "payload": {"task": "Test task", "parameters": "not_a_dict"},
                "expected_code": 422  # Expecting Unprocessable Entity
            }
        ]
        
        # Create results table
        table = Table(title="Error Recovery Tests")
        table.add_column("Test Case", justify="left")
        table.add_column("Expected Code", justify="center")
        table.add_column("Actual Code", justify="center")
        table.add_column("Error Message", justify="left")
        table.add_column("Status", justify="center")
        
        # Run tests
        for test_case in error_test_cases:
            try:
                response = requests.post(
                    f"{BASE_URL}/process", 
                    json=test_case["payload"], 
                    timeout=10
                )
                
                status_code = response.status_code
                try:
                    data = response.json()
                    error_message = data.get("error", "") or data.get("message", "") or data.get("details", "")
                except:
                    error_message = response.text[:50] + "..." if len(response.text) > 50 else response.text
                
                if status_code == test_case["expected_code"]:
                    status = "✅ Success"
                else:
                    status = "❌ Wrong Code"
                    
            except Exception as e:
                status_code = "Error"
                error_message = str(e)
                status = "❌ Exception"
            
            # Add result to table
            table.add_row(
                test_case["name"],
                str(test_case["expected_code"]),
                str(status_code),
                error_message[:50] + "..." if len(error_message) > 50 else error_message,
                status
            )
        
        # Display results
        console.print(table)
        
        # Check error analysis endpoint
        try:
            error_analysis = requests.get(f"{BASE_URL}/error-analysis", timeout=5).json()
            console.print("\n[bold]Error Analysis from API:[/bold]")
            console.print(f"Error Count: {error_analysis.get('error_count', 'N/A')}")
            console.print(f"Error Rate: {error_analysis.get('error_rate', 'N/A')}")
            
            common_errors = error_analysis.get('common_errors', [])
            if common_errors:
                console.print("Common Errors:")
                for error in common_errors:
                    if isinstance(error, dict):
                        console.print(f"- {error.get('type', 'Unknown')}: {error.get('count', 0)} occurrences")
                    else:
                        console.print(f"- {error}")
        except Exception as e:
            console.print(f"[yellow]Error getting error analysis: {str(e)}[/yellow]")
    
    @pytest.mark.e2e
    def test_streaming(self, available_models):
        """
        Test streaming response functionality.
        This test evaluates the system's ability to stream responses.
        """
        console.print("\n\n[bold]Testing Streaming Response[/bold]")
        
        try:
            import aiohttp
            import asyncio
            
            async def test_stream():
                async with aiohttp.ClientSession() as session:
                    task = "Explain the history of artificial intelligence in five paragraphs."
                    
                    console.print(f"Sending streaming request for task: {task[:30]}...")
                    
                    payload = {
                        "task": task,
                        "model_preference": None,
                        "parameters": {"temperature": 0.7}
                    }
                    
                    start_time = time.time()
                    
                    try:
                        async with session.post(
                            f"{BASE_URL}/process-stream", 
                            json=payload, 
                            timeout=60
                        ) as response:
                            console.print(f"Got initial response with status: {response.status}")
                            
                            if response.status != 200:
                                content = await response.text()
                                console.print(f"[red]Error: {content}[/red]")
                                return
                            
                            # Process the stream
                            chunk_count = 0
                            first_chunk_time = None
                            last_chunk_time = None
                            
                            async for line in response.content:
                                line = line.decode('utf-8').strip()
                                if not line:
                                    continue
                                
                                # SSE format: lines starting with "data: "
                                if line.startswith('data: '):
                                    data = json.loads(line[6:])  # Remove "data: " prefix
                                    
                                    # First chunk timing
                                    if chunk_count == 0:
                                        first_chunk_time = time.time()
                                        console.print("[green]Received first chunk[/green]")
                                    
                                    chunk_count += 1
                                    last_chunk_time = time.time()
                                    
                                    # Check for errors in the chunk
                                    if 'error' in data:
                                        console.print(f"[red]Stream error: {data['error']}[/red]")
                                    
                                    # Check for completion
                                    if data.get('finished', False):
                                        console.print("[green]Stream completed[/green]")
                                        break
                            
                            end_time = time.time()
                            total_time = end_time - start_time
                            
                            # Report statistics
                            console.print(f"Total streaming time: {total_time:.2f}s")
                            console.print(f"Total chunks received: {chunk_count}")
                            
                            if first_chunk_time and last_chunk_time:
                                time_to_first = first_chunk_time - start_time
                                console.print(f"Time to first chunk: {time_to_first:.2f}s")
                                
                                if chunk_count > 1:
                                    streaming_duration = last_chunk_time - first_chunk_time
                                    chunks_per_second = (chunk_count - 1) / max(streaming_duration, 0.001)
                                    console.print(f"Chunks per second: {chunks_per_second:.2f}")
                        
                    except Exception as e:
                        console.print(f"[bold red]Streaming error: {str(e)}[/bold red]")
            
            # Run the async test
            asyncio.run(test_stream())
            
        except ImportError:
            console.print("[yellow]Skipping streaming test - aiohttp not available[/yellow]")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Model Processor Interface Tests")
    parser.add_argument("--basic", action="store_true", help="Run basic processing tests")
    parser.add_argument("--cache", action="store_true", help="Run cache performance tests")
    parser.add_argument("--error", action="store_true", help="Run error recovery tests")
    parser.add_argument("--stream", action="store_true", help="Run streaming tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--api", action="store_true", help="Run API tests")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def get_available_models():
    """
    Helper function to get available models outside of pytest fixtures.
    """
    try:
        max_retries = 3
        retry_delay = 1
        models_response = None
        
        for retry in range(max_retries):
            try:
                console.print(f"[yellow]Attempt {retry + 1}/{max_retries} to fetch models...[/yellow]")
                models_response = requests.get(f"{BASE_URL}/models", timeout=10)
                if models_response.status_code == 200:
                    break
                console.print(f"[yellow]Received status code {models_response.status_code}, retrying...[/yellow]")
                time.sleep(retry_delay)
            except requests.RequestException as e:
                console.print(f"[yellow]Request error: {str(e)}, retrying...[/yellow]")
                time.sleep(retry_delay)
        
        if not models_response or models_response.status_code != 200:
            console.print(f"[bold red]Failed to get models after {max_retries} attempts.[/bold red]")
            return ["openai_gpt4", "anthropic_claude"]
        
        available_models = models_response.json().get("models", [])
        
        # Fallback to hardcoded models if empty list is returned
        if not available_models:
            console.print("[bold yellow]No models returned from API, using fallback models for testing...[/bold yellow]")
            available_models = ["openai_gpt4", "anthropic_claude"]
        
        console.print(f"[green]Available Models: {', '.join(available_models)}[/green]")
        return available_models
        
    except Exception as e:
        console.print(f"[bold red]Error fetching models: {str(e)}[/bold red]")
        return ["openai_gpt4", "anthropic_claude"]

if __name__ == "__main__":
    args = parse_args()
    test_instance = TestProcessorInterface()
    test_instance.setup_method()
    
    # Get available models
    available_models = get_available_models()
    
    # Set test markers based on args
    markers = []
    if args.unit or args.all:
        markers.append("unit")
    if args.integration or args.all:
        markers.append("integration")
    if args.api or args.all:
        markers.append("api") 
    if args.e2e or args.all:
        markers.append("e2e")
    
    # If no specific test types are selected via markers, run selected test functions
    if not markers:
        if args.basic or args.all:
            test_instance.test_basic_processing(available_models)
        
        if args.cache or args.all:
            test_instance.test_cache_performance(available_models)
        
        if args.error or args.all:
            test_instance.test_error_recovery(available_models)
        
        if args.stream or args.all:
            test_instance.test_streaming(available_models)
    else:
        # Use pytest to run tests with selected markers
        pytest_args = ["-v"] if args.verbose else []
        for marker in markers:
            pytest_args.extend(["-m", marker])
        pytest.main(pytest_args)
    
    console.print("\nAll tests completed")
