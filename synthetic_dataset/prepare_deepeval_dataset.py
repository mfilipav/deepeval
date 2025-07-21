from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase
import json
from qa_bot import QABot

DATASET_PATH = "dataset_mf.json"

def convert_context_dict_to_string(input_path, output_path):

    # Read the original dataset
    with open(input_path, "r") as f:
        data = json.load(f)

    # Convert context field from dict to string for each item
    for item in data:
        if "context" in item and isinstance(item["context"], dict):
            # Convert dict to a formatted string
            context_str = ", ".join([f"{k}: {v}" for k, v in item["context"].items()])
            item["context"] = [context_str]

    # Write the modified dataset to a new file
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# convert_context_dict_to_string("dataset_mf_w_context_dict.json, DATASET_PATH")

def generate_test_cases(llm_app) -> EvaluationDataset:
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(
        file_path=DATASET_PATH,
        input_key_name="question",
        actual_output_key_name="actual_output",
        expected_output_key_name="answer",
        context_key_name="context",
    )
    # for item in dataset.goldens:
    #     print("\n", item)


    for golden in dataset.goldens:
        output = llm_app.answer(golden.input, str(golden.context))
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=output,
            expected_output=golden.expected_output,
            context=golden.context,
        )
        print(f"input: {test_case.input}\n  --> output: {test_case.actual_output}\n")
        dataset.add_test_case(test_case)
    return dataset

llm_app = QABot(model="gpt-4.1-mini")
dataset = generate_test_cases(llm_app=llm_app)
dataset.save_as(
    file_type="json", 
    directory=".",
    file_name="dataset_mf_test_cases",
    include_test_cases=True
)