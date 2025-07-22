from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

MODEL = "gpt-4.1-mini"
DATASET_FILE_PATH = "dataset_mf_test_cases.json"
# dataset_file_path = "./synthetic_dataset/dataset_mf_test_cases.json"

dataset = EvaluationDataset()

# Get test cases from JSON
dataset.add_test_cases_from_json_file(
    # file_path is the absolute path to you .json file
    file_path=DATASET_FILE_PATH,
    input_key_name="input",
    actual_output_key_name="actual_output",
    expected_output_key_name="expected_output",
    context_key_name="context",
    retrieval_context_key_name="context"
)

# test_case = LLMTestCase(
#     input="What if these shoes don't fit?",
#     # Replace this with the output from your LLM app
#     actual_output="We offer a 30-day full refund at no extra cost."
# )
test_cases = dataset.test_cases
print(f"Loaded {len(test_cases)} test cases from dataset_mf_test_cases.json")
print(f"First test case: {test_cases[0].retrieval_context}")

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=MODEL,
    include_reason=True,
    verbose_mode=True,
)
# metric = FaithfulnessMetric(
#     threshold=0.7,
#     model=MODEL,
#     include_reason=True,
#     verbose_mode=True,
# )

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

evaluate(test_cases=test_cases, metrics=[metric])

