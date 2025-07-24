from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval, MEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.models import GPTModel
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig

MODEL = "gpt-4.1-mini"
# DATASET_FILE_PATH = "dataset_mf_test_cases.json"
DATASET_FILE_PATH = "./synthetic_dataset/dataset_mf_test_cases.json"


dataset = EvaluationDataset()

# Get test cases from JSON
dataset.add_test_cases_from_json_file(
    # file_path is the absolute path to you .json file
    file_path=DATASET_FILE_PATH,
    input_key_name="input",
    actual_output_key_name="actual_output",
    expected_output_key_name="expected_output",
    context_key_name="context",
)

# test_case = LLMTestCase(
#     input="What if these shoes don't fit?",
#     # Replace this with the output from your LLM app
#     actual_output="We offer a 30-day full refund at no extra cost."
# )
test_cases = dataset.test_cases
print(f"Loaded {len(test_cases)} test cases from dataset_mf_test_cases.json")
print(f"First test case: {test_cases[10]}")

metric_answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=MODEL,
    include_reason=True,
    verbose_mode=True,
)
metric_faithfullness = FaithfulnessMetric(
    threshold=0.7,
    model=MODEL,
    include_reason=True,
    verbose_mode=True,
)
metric_geval = GEval(
    name="Correctness",
    evaluation_steps=[
        "Read the 'input' and the 'actual_output'",
        "Check whether the facts in 'actual_output' contradicts any facts in 'expected_output' and user data 'context'",
        "Penalize if the 'actual_output' contradicts any user data in the context.",
        "Vague or polite language, and contradicting OPINIONS are OK"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        # LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.CONTEXT,
    ],
    model=GPTModel(
        model_name=MODEL,
        temperature=0.0,
    ),
    threshold=0.6,
    strict_mode=False,
    async_mode=True,
    verbose_mode=True,
)
metric_meval = MEval(
    name="Usefullness",
    criteria="Does `actual_output` answer user's question (`input`), are the facts in the answer relevant to the user question, are they factually correct, and do they not contradict the facts in the context? If no context is provided, automatically assign a score of 0.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        # LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.CONTEXT,
    ],
    model=GPTModel(
        model_name=MODEL,
        temperature=0.0,
    ),
    threshold=0.6,
    strict_mode=False,
    async_mode=False,
    verbose_mode=True,
)

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

res = evaluate(
    test_cases=[test_cases[0]],
    # test_cases=[test_cases[0], test_cases[10], test_cases[7]],
    metrics=[metric_meval],
    async_config=AsyncConfig(run_async=False),
    display_config=DisplayConfig(
        file_output_dir="./synthetic_dataset/evaluation_logs_xml"
    ),
)

