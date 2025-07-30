#!/usr/bin/env python3
"""
Simple script to demonstrate and profile DeepEval's asyncio parallelization.
This script uses a stub LLM that sleeps for 0.2 seconds to simulate LLM latency.
"""

import asyncio
import time
from typing import List
import json
import matplotlib.pyplot as plt
import numpy as np

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig

LATENCY_SECONDS = 0.2

class StubLLM(DeepEvalBaseLLM):
    """Stub LLM that simulates 0.2s latency instead of real API calls"""
    
    def __init__(self, latency_seconds: float = 0.2):
        self.latency_seconds = latency_seconds
        super().__init__(model_name="stub-llm")
    
    def load_model(self):
        return "stub-model"
    
    def generate(self, prompt: str, *args, **kwargs) -> str:
        """Sync call with simulated latency"""
        time.sleep(self.latency_seconds)
        return f"Response: {prompt}..."
    
    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        """Async call with simulated latency"""
        await asyncio.sleep(self.latency_seconds)
        return f"Response: {prompt}..."
    
    def get_model_name(self) -> str:
        return f"StubLLM-{self.latency_seconds}s"


class SimpleTestMetric(BaseMetric):
    """Simple metric that makes 2 LLM calls per evaluation"""
    
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]
    
    def __init__(self, model: DeepEvalBaseLLM):
        self.model = model
        self.using_native_model = False
        self.threshold = 0.5
        self.async_mode = True
        self.evaluation_model = model.get_model_name()
        self.strict_mode = False
        self.verbose_mode = False
        self.include_reason = False
        
        # Initialize state
        self.score = None
        self.success = None
        self.reason = None
        self.error = None
        self.evaluation_cost = None
        self.verbose_logs = None
        self.skipped = False
    
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Sync evaluation - calls LLM sequentially"""
        # Call 1: Analyze input
        self.model.generate(f"Analyze: {test_case.input}")
        
        # Call 2: Evaluate output  
        self.model.generate(f"Evaluate: {test_case.actual_output}")
        
        self.score = 0.8  # Dummy score
        self.success = self.score >= self.threshold
        self.reason = "Evaluation completed"
        return self.score
    
    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Async evaluation - calls LLM concurrently"""
        # Both calls run concurrently
        call1 = self.model.a_generate(f"Analyze: {test_case.input}")
        call2 = self.model.a_generate(f"Evaluate: {test_case.actual_output}")
        
        # Wait for both to complete
        await asyncio.gather(call1, call2)
        
        self.score = 0.8  # Dummy score
        self.success = self.score >= self.threshold
        self.reason = "Evaluation completed"
        return self.score
    
    def is_successful(self) -> bool:
        return self.success if self.success is not None else False
    
    @property
    def __name__(self):
        return "SimpleTestMetric"


def profiling(concurrency_levels) -> List[dict]:    
    print("\n=== Profiling Analysis ===\n")
    
    stub_llm = StubLLM(latency_seconds=LATENCY_SECONDS)
    
    # Test with different numbers of test cases and metrics
    configurations = [
        # (1, 1),  # 1 test case, 1 metric
        # (2, 2),  # 2 test cases, 2 metrics
        # (20, 5),
        # (5, 20),
        # (100, 1),
        # (100, 20),
        (100, 5),
        (1000, 5),
        (5000, 5)
    ]
    
    results = []
    for num_cases, num_metrics in configurations:        
        test_cases = [
            LLMTestCase(
                input=f"Test question {i+1} {(i%100)*'A'}",
                actual_output=f"Test answer {i+1} {(i%100)*'B'}"
            ) for i in range(num_cases)
        ]
        
        metrics = [SimpleTestMetric(stub_llm) for _ in range(num_metrics)]
        # print(f"    DEBUG: test cases: {test_cases}")
        # Test various concurrency levels
        for max_concurrent in concurrency_levels:

            start_time = time.time()
            evaluate(
                test_cases=test_cases,
                metrics=metrics,
                async_config=AsyncConfig(
                    run_async=True,
                    max_concurrent=max_concurrent,
                    throttle_value=0.01
                ),
                display_config=DisplayConfig(show_indicator=False, print_results=False)
            )
            end_time = time.time()

            total_time = end_time - start_time
            sequential_estimate = num_cases * num_metrics * 2 * LATENCY_SECONDS
            speedup = sequential_estimate / total_time

            theoretical_min = 2 * LATENCY_SECONDS  # 2 LLM calls per metric, minimum possible time
            
            # Store result
            result = {
                "num_cases": num_cases,
                "num_metrics": num_metrics,
                "max_concurrent": max_concurrent,
                "total_time": total_time,
                "sequential_estimate": sequential_estimate,
                "speedup": speedup,
                "efficiency_concurrent": speedup / max_concurrent,
                "efficiency_theoretical": theoretical_min / total_time,
                "theoretical_min": theoretical_min
            }
            results.append(result)
            
            print(f"Configuration: {num_cases} test cases x {num_metrics} metrics at max_concurrent={max_concurrent}")
            print(f"  Total time:                          {total_time:.2f}s")
            print(f"  Speedup over sequential:             {speedup:.2f}x")
            print(f"  Efficiency (speedup/max_concurrent): {speedup / max_concurrent:.2f}")
            print(f"  Efficiency (theoret_min/total_time): {theoretical_min / total_time:.2f}")
            print("\n")
    
    # Write results to JSON file
    json_filename = f"profiling/profile_asyncio_{time.strftime('%Y%m%d_%H%M%S')}_results.json"
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_filename}")
        
    return results


def create_timing_plot(results: List[dict], concurrency_levels: List[int]):
    """Create matplotlib plot showing total time elapsed for each combination"""
    MAX_CONC = 100
    # Group results by configuration for better visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data for plotting
    configurations = []
    times_by_concurrent = {concurrent: [] for concurrent in concurrency_levels}

    for result in results:
        config_label = f"{result['num_cases']}c x {result['num_metrics']}m"
        if config_label not in configurations:
            configurations.append(config_label)
        times_by_concurrent[result['max_concurrent']].append(result['total_time'])
    
    # Plot 1: Total time for each configuration by concurrency level
    x = np.arange(len(configurations))
    width = 0.2
    
    for idx, concurrent in enumerate(concurrency_levels):
        offset = (idx - (len(concurrency_levels) - 1) / 2) * width
        ax1.bar(x + offset, times_by_concurrent[concurrent], width, label=f'max_concurrent={concurrent}', alpha=0.8)

    ax1.set_xlabel('Configuration (C cases x M metrics)')
    ax1.set_ylabel('Total Time (seconds)')
    ax1.set_title('Total Execution Time by number of Cases and Metrics, and Concurrency')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configurations, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup vs Max Concurrent
    speedups_by_concurrent = {}
    
    for concurrent in concurrency_levels:
        concurrent_results = [r for r in results if r['max_concurrent'] == concurrent]
        avg_speedup = np.mean([r['speedup'] for r in concurrent_results])
        speedups_by_concurrent[concurrent] = avg_speedup
    
    ax2.plot(list(speedups_by_concurrent.keys()), list(speedups_by_concurrent.values()), 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Max Concurrent')
    ax2.set_ylabel('Average Speedup')
    ax2.set_title('Average Speedup vs Concurrency Level')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Heatmap of total time by num_cases and num_metrics for max_concurrent=20
    cases_metrics_combos = [(r['num_cases'], r['num_metrics']) for r in results if r['max_concurrent'] == MAX_CONC]
    unique_cases = sorted(set([combo[0] for combo in cases_metrics_combos]))
    unique_metrics = sorted(set([combo[1] for combo in cases_metrics_combos]))
    
    heatmap_data = np.zeros((len(unique_metrics), len(unique_cases)))
    for result in results:
        if result['max_concurrent'] == MAX_CONC:
            case_idx = unique_cases.index(result['num_cases'])
            metric_idx = unique_metrics.index(result['num_metrics'])
            heatmap_data[metric_idx, case_idx] = result['total_time']
    
    im = ax3.imshow(heatmap_data, cmap='viridis', aspect='auto')
    ax3.set_xlabel('Number of Test Cases')
    ax3.set_ylabel('Number of Metrics')
    ax3.set_title(f'Total Time Heatmap (max_concurrent={MAX_CONC})')
    ax3.set_xticks(range(len(unique_cases)))
    ax3.set_xticklabels(unique_cases)
    ax3.set_yticks(range(len(unique_metrics)))
    ax3.set_yticklabels(unique_metrics)
    
    # Add text annotations to heatmap
    for i in range(len(unique_metrics)):
        for j in range(len(unique_cases)):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.1f}s',
                           ha="center", va="center", color="white", fontweight='bold')
    
    plt.colorbar(im, ax=ax3, label='Total Time (seconds)')
    
    # Plot 4: Efficiency comparison
    efficiency_data = []
    config_labels = []
    
    for result in results:
        if result['max_concurrent'] == MAX_CONC:  # Focus on moderate concurrency
            config_labels.append(f"{result['num_cases']}c x {result['num_metrics']}m")
            efficiency_data.append(result['efficiency_concurrent'])
    
    ax4.bar(range(len(config_labels)), efficiency_data, alpha=0.7, color='orange')
    ax4.set_xlabel('Configuration (C cases x M metrics)')
    ax4.set_ylabel('Efficiency (speedup/max_concurrent)')
    ax4.set_title('Parallelization Efficiency (max_concurrent={MAX_CONC})')
    ax4.set_xticks(range(len(config_labels)))
    ax4.set_xticklabels(config_labels, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"profiling/profile_asyncio_{time.strftime('%Y%m%d_%H%M%S')}_results.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    


if __name__ == "__main__":
    print("DeepEval Asyncio Parallelization Profiler")
    print(f"Using stub LLM with {LATENCY_SECONDS}s simulated latency\n")
    concurrency_levels = [20, 100]
    results = profiling(concurrency_levels=concurrency_levels)
    create_timing_plot(results, concurrency_levels=concurrency_levels)

    # 1. Higher max_concurrent values should reduce execution time
    # 2. Speedup improvements depend on the amount of parallelizable work
    # 3. There's overhead in async coordination, so efficiency may decrease
    # 4. Optimal concurrency depends on the number of concurrent operations