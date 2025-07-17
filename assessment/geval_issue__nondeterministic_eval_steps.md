## G-Eval scores vary too much: LLM generates different Evaluation Steps from `criteria` string
```python
metric = GEval(
    name="Correctness",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    criteria="Determine whether the actual output is factually correct based on expected output.",
    # evaluation_steps=[
    #     "Read the Input prompt and the Actual Output.",
    #     "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
    #     "You should also heavily penalize omission of detail",
    #     "Vague language, or contradicting OPINIONS, are OK"
    # ],
    # NOTE: you can only provide eval params that are mentioned in criteria or evaluation_steps!
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
        # LLMTestCaseParams.CONTEXT,
    ],
    # rubric=[
    #     Rubric(score_range=(0,2), expected_outcome="Factually incorrect."),
    #     Rubric(score_range=(3,6), expected_outcome="Mostly correct."),
    #     Rubric(score_range=(7,9), expected_outcome="Correct but missing minor details."),
    #     Rubric(score_range=(10,10), expected_outcome="100% correct."),
    # ],
    threshold=0.6,
    strict_mode=False,
    async_mode=True,
    verbose_mode=True,
)
test_case = LLMTestCase(
    input="Where and when was Einstein born?",
    actual_output="Einstein was born in Germany on 20th March 1879.",
    expected_output="Einstein was born in Germany on 14th March 1879.",
    # context=["Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time"],
)
```

success example, high score:
    Criteria:
    Determine whether the actual output is factually correct based on expected output. 
    
    Evaluation Steps:
    [
        "Review the Input to understand the context and requirements.",
        "Compare the Actual Output to the Expected Output for factual correctness.",
        "Identify any factual discrepancies between the Actual Output and Expected Output based on the Input."
    ]
    
    Score: 0.747147240978186
    Reason: The Actual Output correctly identifies Germany as Einstein's birthplace, aligning with the Expected Output. However, it provides an incorrect birth date, stating 20th March 1879 instead of the correct 14th March 1879. This factual discrepancy regarding the date reduces the score, despite the correct country.

success:
    Criteria:
    Determine whether the actual output is factually correct based on expected output. 
    
    Evaluation Steps:
    [
        "Review the Input to understand the context and requirements of the task.",
        "Compare the Actual Output to the Expected Output, checking for factual consistency and accuracy.",
        "Identify any discrepancies or errors between the Actual Output and Expected Output.",
        "Conclude if the Actual Output is factually correct based on its alignment with the Expected Output."
    ]
    Score: 0.6902437039335128
    Reason: The Actual Output correctly identifies Germany as Einstein's birthplace, aligning with the Expected Output. However, it provides the incorrect birth date, stating 20th March instead of the correct 14th March 1879. This factual inaccuracy regarding the date results in a deduction, but the response is otherwise mostly accurate.


failure, high score, almost succeeded:

    E           AssertionError: Metrics: Correctness [GEval] (score: 0.5763835194595683, threshold: 0.6, strict: False, error: None, reason: The actual output correctly identifies Germany as Einstein's birthplace, aligning with the expected output. However, it provides the incorrect birth date, stating 20th March instead of the correct 14th March 1879. This factual inaccuracy regarding the date reduces the alignment with the evaluation steps.)

    E           self.verbose_logs:
    E           Criteria:
    E           Determine whether the actual output is factually correct based on expected output. 
    E            
    E           Evaluation Steps:
    E           [
    E               "Review the input to understand the context and requirements.",
    E               "Compare the actual output to the expected output for factual consistency.",
    E               "Verify that all facts presented in the actual output align with the expected output.",
    E               "Conclude whether the actual output is factually correct based on this comparison."
    E           ] 
    E            
    E           Rubric:
    E           None 
    E            
    E           Score: 0.5763835194595683 
    E           Cost: 0.002072 failed.


failure, low score:

    AssertionError: Metrics: Correctness [GEval] (score: 0.3211879220850647, threshold: 0.6, strict: False, error: None, reason: The Actual Output correctly identifies Germany as Einstein's birthplace, aligning with the Expected Output. However, it provides the incorrect birth date, stating 20th March instead of the correct 14th March 1879. This factual inaccuracy significantly reduces alignment with the evaluation steps.)

    E           self.verbose_logs:
    E           Criteria:
    E           Determine whether the actual output is factually correct based on expected output. 
    E            
    E           Evaluation Steps:
    E           [
    E               "Read the Input to understand the context and requirements.",
    E               "Compare the Actual Output to the Expected Output for factual alignment.",
    E               "Verify if the Actual Output contains any inaccuracies or deviations from the Expected Output.",
    E               "Determine if the Actual Output is factually correct based on the Expected Output."
    E           ] 
    E           Score: 0.3211879220850647
    E           Cost: 0.002048 failed.


### LLM struggles to assign weights to incorrect D-M-Y items
Hypothesis: Germany is correct. However, LLM might struggle to decide how much weight to assign to incorrect Day in YearMonthDay tuple.

If criteria where changed to penalize incorrect Day, most of the scores are 0.2-0.4:

        criteria="Determine whether the actual output is factually correct based on expected output. If the day is incorrect, you should heavily penalize it, but if the month or year is incorrect, you should not penalize it as much.",


## Explicit evaluation steps result in varying scores

Does the low score remain 0.321 if evaluation steps are explicitly set? Re-use eval steps from "failure, low score" case above.

Result: no, scores fluctuate between 0.3 to 0.55

```python
    metric = GEval(
        name="Correctness",
        # NOTE: you can only provide either criteria or evaluation_steps, and not both
        criteria=None,
        evaluation_steps=[
            "Read the Input to understand the context and requirements.",
            "Compare the Actual Output to the Expected Output for factual alignment.",
            "Verify if the Actual Output contains any inaccuracies or deviations from the Expected Output.",
            "Determine if the Actual Output is factually correct based on the Expected Output."
        ],
        # NOTE: you can only provide eval params that are mentioned in criteria or evaluation_steps!
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            # LLMTestCaseParams.CONTEXT,
        ],
        # rubric=[
        #     Rubric(score_range=(0,2), expected_outcome="Factually incorrect."),
        #     Rubric(score_range=(3,6), expected_outcome="Mostly correct."),
        #     Rubric(score_range=(7,9), expected_outcome="Correct but missing minor details."),
        #     Rubric(score_range=(10,10), expected_outcome="100% correct."),
        # ],
        threshold=0.6,
        strict_mode=False,
        async_mode=True,
        verbose_mode=True,
    )
    test_case = LLMTestCase(
        input="Where and when was Einstein born?",
        actual_output="Einstein was born in Germany on 20th March 1879.",
        expected_output="Einstein was born in Germany on 14th March 1879.",
        # context=["Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time"],
    )
```

low score example:

    E           AssertionError: Metrics: Correctness [GEval] (score: 0.38848666300135504, threshold: 0.6, strict: False, error: None, reason:
    The Actual Output correctly identifies Germany as Einstein's birthplace, aligning with the Expected Output.
    However, it provides an incorrect birth date (20th March instead of 14th March 1879), which is a significant factual error.
    This major inaccuracy substantially reduces alignment with the evaluation steps.)
    E           self.verbose_logs:
    E           Criteria:
    E           None 
    E            
    E           Evaluation Steps:
    E           [
    E               "Read the Input to understand the context and requirements.",
    E               "Compare the Actual Output to the Expected Output for factual alignment.",
    E               "Verify if the Actual Output contains any inaccuracies or deviations from the Expected Output.",
    E               "Determine if the Actual Output is factually correct based on the Expected Output."
    E           ] 
    E           Score: 0.38848666300135504
    E           Cost: 0.001242 failed.


high score:

    E           AssertionError: Metrics: Correctness [GEval] (score: 0.515439315947518, threshold: 0.6, strict: False, error: None, reason: 
    The Actual Output correctly identifies Germany as Einstein's birthplace, aligning with the Expected Output.
    However, it provides an incorrect birth date (20th March instead of 14th March 1879), which is a factual inaccuracy.
    This partial correctness warrants a moderate score.)
    E           self.verbose_logs:
    E           Criteria:
    E           None 
    E            
    E           Evaluation Steps:
    E           [
    E               "Read the Input to understand the context and requirements.",
    E               "Compare the Actual Output to the Expected Output for factual alignment.",
    E               "Verify if the Actual Output contains any inaccuracies or deviations from the Expected Output.",
    E               "Determine if the Actual Output is factually correct based on the Expected Output."
    E           ] 
    E            
    E           Score: 0.515439315947518 
    E           Cost: 0.00121 failed.


## How much does it cost for CoT to generate evaluation steps from `criteria` string?

cost comparison:

    criteria --CoT LLM generation--> eval steps:    0.0021
    explicit eval steps:                            0.0012

Result: using CoT to generate eval steps costs 75% more than explicitly setting eval steps.
However, the domain has to be known and fully understood to set eval steps ahead.



## Increasing LLM temperature guarantees that tests succeed and reasoning is more creative
Default temp is 0.0:
```python
class GPTModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: "gpt-4.1",
        temperature: float = 0,
    )
```


With 2.0 temp the LLMaaj always passes the test and its judgement is more "creative".

    Criteria:
    Determine whether the actual output is factually correct based on expected output. 
    
    Evaluation Steps:
    [
        "Review the Input to understand the context and requirements.",
        "Compare the Actual Output to the Expected Output for factual consistency.",
        "Identify any discrepancies or inaccuracies between the Actual Output and the Expected Output.",
        "Decide if the Actual Output is factually correct based on the Expected Output."
    ] 

    Score: 0.7059399850537137 
    Reason: The response correctly identifies the country (Germany) as per the expectation, but provides an incorrect birth date by six days: it says 20th March instead of the correct 14th March as listed in the Expected Output. This discrepancy prevents full factual correctness, although the Actual Output closely matches otherwise.    
    Evaluation Cost: 0.002086


Another example with a verbose answer:

    Criteria:
    Determine whether the actual output is factually correct based on expected output. 
 
    Evaluation Steps:
    [
        "Read and understand the Input to establish the context.",
        "Compare the Expected Output to the Actual Output for factual accuracy.",
        "Check if every fact in the Actual Output matches the facts in the Expected Output.",
        "Decide if the Actual Output is factually correct based on this comparison."
    ] 
    
    Rubric:
    None 
    
    Score: 0.6396816607334018 
    
    Reason: The actual output correctly states that Einstein was born in Germany, following the expected country's detail. However, the birthdate provided ('20th March 1879') does not match the expected date ('14th March 1879'); therefore, not all facts are accurate per step 3, but the majority of the information matches, warranting some credit.
    
    Evaluation Cost: 0.002184


## what if Einstein's birthplace is also misjudged?
```python
    test_case = LLMTestCase(
        input="Where and when was Einstein born?",
        actual_output="Einstein was born in Poland on 20th March 1879.",
        expected_output="Einstein was born in Germany on 14th March 1879.",
    )
```

Scores converge at around 0.1. Quite stable.


