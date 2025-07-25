import os
import time
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.evaluate.types import EvaluationResult
from synthetic_dataset.utils import (
    visualize_eval_results,
    print_eval_results,
    results_to_df
)
from synthetic_dataset.eval_metrics import (
    metric_answer_relevancy,
    # metric_faithfullness,
    # metric_geval,
    metric_meval,
    # metric_latency
)

# WARNING: change value when experimenting with different combinations of:
#   metrics, test cases or evaluator parameters
EXPERIMENT_NAME = "simple_meval_answer_relevancy"

EXPERIMENTS_DIR = "./synthetic_dataset/experiments/"
DATASET_FILE_PATH = "./synthetic_dataset/dataset_mf_test_cases.json"

# 1. Prepare test cases from the dataset
dataset = EvaluationDataset()
dataset.add_test_cases_from_json_file(
    # file_path is the absolute path to you .json file
    file_path=DATASET_FILE_PATH,
    input_key_name="input",
    actual_output_key_name="actual_output",
    expected_output_key_name="expected_output",
    context_key_name="context",
)
test_cases = dataset.test_cases
print(f"Loaded {len(test_cases)} test cases from {DATASET_FILE_PATH}")
# print(f"First test case: {test_cases[10]}")


# 2. Run evaluation experiment
experiment_result_path = f"{EXPERIMENTS_DIR}{EXPERIMENT_NAME}"
res: EvaluationResult = evaluate(
    # test_cases=test_cases,
    test_cases=[test_cases[0], test_cases[10]],
    # test_cases=[test_cases[0], test_cases[10], test_cases[7]],
    metrics=[metric_meval, metric_answer_relevancy],
    async_config=AsyncConfig(run_async=True),
    display_config=DisplayConfig(
        file_output_dir=experiment_result_path
    ),
)
ts = time.strftime("%Y%m%d_%H%M%S")


# 3. Save eval results to JSON file and plot a figure
df = results_to_df(res, json_file_path=f"{experiment_result_path}/test_run_{ts}_results.json")
print("\nPandas DataFrame:")
print(df.to_string(index=False))

experiment_figure_path = os.path.join(
    experiment_result_path, f"test_run_{ts}_figure.png")
visualize_eval_results(df, experiment_figure_path)