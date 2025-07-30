#!/usr/bin/env python3
"""
Script to profile DeepEval's asyncio parallelization performance.
This script uses a stub LLM that sleeps for 0.2 seconds to simulate LLM latency
without making actual API calls.
"""

import asyncio
import time
from typing import List, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig


@dataclass
class ProfileResult:
    """Container for profiling results"""
    test_cases: int
    metrics: int
    max_concurrent: int
    total_time: float
    sequential_time_estimate: float
    speedup: float
    async_mode: bool


class StubLLM(DeepEvalBaseLLM):
    """
    Stub LLM that simulates API latency by sleeping for a fixed duration
    instead of making actual LLM calls.
    """
    
    def __init__(self, latency_seconds: float = 0.2):
        self.latency_seconds = latency_seconds
        super().__init__(model_name="stub-llm")
    
    def load_model(self):
        return "stub-model"
    
    def generate(self, prompt: str, *args, **kwargs) -> str:
        """Synchronous generation with simulated latency"""
        time.sleep(self.latency_seconds)
        return f"Generated response for: {prompt[:50]}..."
    
    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        """Asynchronous generation with simulated latency"""
        await asyncio.sleep(self.latency_seconds)
        return f"Generated response for: {prompt[:50]}..."
    
    def get_model_name(self) -> str:
        return f"StubLLM-{self.latency_seconds}s"


class TestMetric(BaseMetric):
    """
    Simple test metric that makes multiple LLM calls to test parallelization.
    Each metric evaluation makes `num_llm_calls` to the model.
    """
    
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]
    
    def __init__(self, 
                 model: DeepEvalBaseLLM,
                 num_llm_calls: int = 3,
                 threshold: float = 0.5,
                 async_mode: bool = True):
        self.model = model
        self.using_native_model = False
        self.num_llm_calls = num_llm_calls
        self.threshold = threshold
        self.async_mode = async_mode
        self.evaluation_model = model.get_model_name()
        self.strict_mode = False
        self.verbose_mode = False
        self.include_reason = False
        
        # Initialize metric state
        self.score = None
        self.success = None
        self.reason = None
        self.error = None
        self.evaluation_cost = None
        self.verbose_logs = None
        self.skipped = False
    
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Synchronous measurement"""
        start_time = time.time()
        
        # Simulate multiple LLM calls that would happen in real metrics
        responses = []
        for i in range(self.num_llm_calls):
            prompt = f"Evaluate step {i+1}: {test_case.input} -> {test_case.actual_output}"
            response = self.model.generate(prompt)
            responses.append(response)
        
        # Simulate score calculation
        self.score = np.random.uniform(0.3, 1.0)
        self.success = self.score >= self.threshold
        self.reason = f"Evaluation completed with {len(responses)} LLM calls"
        
        end_time = time.time()
        print(f"  Metric completed in {end_time - start_time:.2f}s (sync)")
        
        return self.score
    
    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Asynchronous measurement with concurrent LLM calls"""
        start_time = time.time()
        
        # Create concurrent LLM calls
        tasks = []
        for i in range(self.num_llm_calls):
            prompt = f"Evaluate step {i+1}: {test_case.input} -> {test_case.actual_output}"
            tasks.append(self.model.a_generate(prompt))
        
        # Wait for all LLM calls to complete concurrently
        responses = await asyncio.gather(*tasks)
        
        # Simulate score calculation
        self.score = np.random.uniform(0.3, 1.0)
        self.success = self.score >= self.threshold
        self.reason = f"Evaluation completed with {len(responses)} LLM calls"
        
        end_time = time.time()
        print(f"  Metric completed in {end_time - start_time:.2f}s (async)")
        
        return self.score
    
    def is_successful(self) -> bool:
        return self.success if self.success is not None else False
    
    @property
    def __name__(self):
        return f"TestMetric-{self.num_llm_calls}calls"


def create_test_cases(num_cases: int) -> List[LLMTestCase]:
    """Create test cases for profiling"""
    test_cases = []
    for i in range(num_cases):
        test_case = LLMTestCase(
            input=f"Test input {i+1}: What is the capital of France?",
            actual_output=f"Test output {i+1}: The capital of France is Paris.",
            expected_output="The capital of France is Paris."
        )
        test_cases.append(test_case)
    return test_cases


def create_test_metrics(stub_llm: StubLLM, num_metrics: int, num_llm_calls: int = 3) -> List[TestMetric]:
    """Create test metrics for profiling"""
    metrics = []
    for i in range(num_metrics):
        metric = TestMetric(
            model=stub_llm,
            num_llm_calls=num_llm_calls,
            async_mode=True
        )
        metrics.append(metric)
    return metrics


def run_profile(test_cases: List[LLMTestCase], 
                metrics: List[TestMetric], 
                max_concurrent: int,
                async_mode: bool = True) -> ProfileResult:
    """Run a single profiling test"""
    
    print(f"\n=== Profiling: {len(test_cases)} test cases, {len(metrics)} metrics, "
          f"max_concurrent={max_concurrent}, async_mode={async_mode} ===")
    
    start_time = time.time()
    
    # Configure evaluation
    async_config = AsyncConfig(
        run_async=async_mode,
        max_concurrent=max_concurrent,
        throttle_value=0  # No throttling for cleaner profiling
    )
    display_config = DisplayConfig(
        show_indicator=False,  # Disable progress bars for cleaner output
        print_results=False,
        file_output_dir=None
    )
    
    # Run evaluation
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=async_config,
        display_config=display_config
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate theoretical sequential time
    # Each test case * each metric * num_llm_calls * latency per call
    sequential_time_estimate = (len(test_cases) * len(metrics) * 
                              metrics[0].num_llm_calls * 0.2)
    
    speedup = sequential_time_estimate / total_time if total_time > 0 else 0
    
    profile_result = ProfileResult(
        test_cases=len(test_cases),
        metrics=len(metrics),
        max_concurrent=max_concurrent,
        total_time=total_time,
        sequential_time_estimate=sequential_time_estimate,
        speedup=speedup,
        async_mode=async_mode
    )
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Sequential estimate: {sequential_time_estimate:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    return profile_result


def run_comprehensive_profile():
    """Run comprehensive profiling across different configurations"""
    
    print("Starting DeepEval Asyncio Parallelization Profiling...")
    print("=" * 60)
    
    # Create stub LLM with 0.2s latency
    stub_llm = StubLLM(latency_seconds=0.2)
    
    # Profile configurations
    test_cases_counts = [5, 10, 20]
    metrics_counts = [2, 3]
    max_concurrent_values = [1, 5, 10, 20]
    
    results = []
    
    for num_test_cases in test_cases_counts:
        for num_metrics in metrics_counts:
            test_cases = create_test_cases(num_test_cases)
            metrics = create_test_metrics(stub_llm, num_metrics, num_llm_calls=3)
            
            for max_concurrent in max_concurrent_values:
                # Test async mode
                result = run_profile(test_cases, metrics, max_concurrent, async_mode=True)
                results.append(result)
                
                # Test sync mode for comparison (only for smaller configurations)
                if num_test_cases <= 5 and num_metrics <= 2 and max_concurrent == 1:
                    sync_result = run_profile(test_cases, metrics, max_concurrent, async_mode=False)
                    results.append(sync_result)
    
    return results


def analyze_results(results: List[ProfileResult]):
    """Analyze and visualize profiling results"""
    
    print("\n" + "=" * 60)
    print("PROFILING RESULTS ANALYSIS")
    print("=" * 60)
    
    # Group results by configuration
    async_results = [r for r in results if r.async_mode]
    sync_results = [r for r in results if not r.async_mode]
    
    print(f"\nAsync Results ({len(async_results)} configurations):")
    print("-" * 50)
    for result in async_results:
        print(f"Cases: {result.test_cases:2d}, Metrics: {result.metrics}, "
              f"Concurrent: {result.max_concurrent:2d}, "
              f"Time: {result.total_time:5.2f}s, Speedup: {result.speedup:5.2f}x")
    
    if sync_results:
        print(f"\nSync Results ({len(sync_results)} configurations):")
        print("-" * 50)
        for result in sync_results:
            print(f"Cases: {result.test_cases:2d}, Metrics: {result.metrics}, "
                  f"Time: {result.total_time:5.2f}s, Speedup: {result.speedup:5.2f}x")
    
    # Find best configurations
    best_speedup = max(async_results, key=lambda x: x.speedup)
    print(f"\nBest Speedup Configuration:")
    print(f"  {best_speedup.test_cases} test cases, {best_speedup.metrics} metrics, "
          f"max_concurrent={best_speedup.max_concurrent}")
    print(f"  Speedup: {best_speedup.speedup:.2f}x ({best_speedup.total_time:.2f}s vs {best_speedup.sequential_time_estimate:.2f}s)")
    
    # Visualize results
    create_visualizations(async_results)


def create_visualizations(results: List[ProfileResult]):
    """Create visualizations of profiling results"""
    try:
        # Group by test configuration for plotting
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Speedup vs Max Concurrent
        concurrent_values = sorted(set(r.max_concurrent for r in results))
        speedups_by_concurrent = {}
        
        for concurrent in concurrent_values:
            concurrent_results = [r for r in results if r.max_concurrent == concurrent]
            avg_speedup = np.mean([r.speedup for r in concurrent_results])
            speedups_by_concurrent[concurrent] = avg_speedup
        
        ax1.plot(list(speedups_by_concurrent.keys()), list(speedups_by_concurrent.values()), 'bo-')
        ax1.set_xlabel('Max Concurrent')
        ax1.set_ylabel('Average Speedup')
        ax1.set_title('Speedup vs Concurrency Level')
        ax1.grid(True)
        
        # Plot 2: Total Time vs Max Concurrent
        times_by_concurrent = {}
        for concurrent in concurrent_values:
            concurrent_results = [r for r in results if r.max_concurrent == concurrent]
            avg_time = np.mean([r.total_time for r in concurrent_results])
            times_by_concurrent[concurrent] = avg_time
        
        ax2.plot(list(times_by_concurrent.keys()), list(times_by_concurrent.values()), 'ro-')
        ax2.set_xlabel('Max Concurrent')
        ax2.set_ylabel('Average Total Time (s)')
        ax2.set_title('Execution Time vs Concurrency Level')
        ax2.grid(True)
        
        # Plot 3: Speedup vs Total Operations
        operations = [(r.test_cases * r.metrics) for r in results]
        speedups = [r.speedup for r in results]
        ax3.scatter(operations, speedups, alpha=0.6)
        ax3.set_xlabel('Total Operations (test_cases × metrics)')
        ax3.set_ylabel('Speedup')
        ax3.set_title('Speedup vs Total Operations')
        ax3.grid(True)
        
        # Plot 4: Efficiency (speedup/concurrent) vs Max Concurrent
        efficiencies_by_concurrent = {}
        for concurrent in concurrent_values:
            concurrent_results = [r for r in results if r.max_concurrent == concurrent]
            avg_efficiency = np.mean([r.speedup / r.max_concurrent for r in concurrent_results])
            efficiencies_by_concurrent[concurrent] = avg_efficiency
        
        ax4.plot(list(efficiencies_by_concurrent.keys()), list(efficiencies_by_concurrent.values()), 'go-')
        ax4.set_xlabel('Max Concurrent')
        ax4.set_ylabel('Efficiency (Speedup/Concurrent)')
        ax4.set_title('Parallelization Efficiency')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('/Users/modestas/dev/deepeval/asyncio_profiling_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nVisualization saved as: asyncio_profiling_results.png")
        
    except ImportError:
        print("\nMatplotlib not available - skipping visualizations")


def main():
    """Main profiling execution"""
    print("DeepEval Asyncio Parallelization Profiler")
    print("This script profiles the performance of DeepEval's async evaluation")
    print("using a stub LLM that simulates 0.2s latency per call.\n")
    
    # Run comprehensive profiling
    results = run_comprehensive_profile()
    
    # Analyze results
    analyze_results(results)
    
    print("\nProfiling completed!")
    print("\nKey insights:")
    print("- Higher max_concurrent values should reduce total execution time")
    print("- Speedup should increase with more concurrent operations")
    print("- Efficiency may decrease as concurrency increases due to overhead")
    print("- The optimal concurrency level depends on the number of operations")


if __name__ == "__main__":
    main()