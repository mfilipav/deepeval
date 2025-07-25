from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval, MEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel

MODEL = "gpt-4.1-mini"


metric_answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=MODEL,
    include_reason=True,
    verbose_mode=False,
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
    async_mode=True,
    verbose_mode=False,
)
