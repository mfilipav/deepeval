## Synthetic dataset Creation
`./generate_dataset` has files with tabular data (`synthetic_dataset/generate_dataset/final_FedCycleData.json`) and text data (`synthetic_dataset/generate_dataset/filtered_combi_mf.json`). These two datasets were cherry-picked to generate `synthetic_dataset/dataset_mf.json`.

Deepeval goldens were prepared by ingesting `dataset_mf.json` and fixing its context (must be a string, not a dict).

Test cases were prepared by running LLM agent `synthetic_dataset/qa_bot.py::QABot`, populating `actual_output` field with the LLM output, and writing out two files: `dataset_mf_test_cases.json` (see https://deepeval.com/docs/evaluation-datasets#save-dataset) and `dataset_mf_goldens.json`. Goldens have no `actual_output` field.

Example test cases (`LLMTestCase` objects) inside `dataset_mf_test_cases.json`:
```json
    {
        "input": "My vulva has been super itchy",
        "actual_output": "Itchy vulva can be caused by several things like irritation, yeast infections, or allergies. Since it's been bothering you, it’s a good idea to avoid any new soaps or products in that area and keep things clean and dry. If the itching persists, worsens, or you notice other symptoms like unusual discharge or pain, it’s best to see a doctor to get it checked out.",
        "expected_output": "Experiencing an itchy vulva can be really uncomfortable. There are various reasons why this might be happening, such as infections, irritation from products, or skin conditions. It seems that you've been having a lot of sex lately (30 times in 20 days), you might have an STI. Since I can't offer medical advice, it's best to consult with a healthcare professional to figure out the cause and find the right solution.",
        "retrieval_context": null,
        "context": [
            "Age: 22, BMI: 21.9485766758494, EndMenses: 2025-04-18, Height: 66, IntercourseInFertileWindow: 30, LengthofCycle: 27, LengthofMenses: 5, MeanBleedingIntensity: 10.17, MeanCycleLength: 27.17, MeanMensesLength: 5.33, NumberofDaysofIntercourse: 20, StartMenses: 2025-04-14, Today: 2025-04-29, TotalMensesScore: 11, UnusualBleeding: 1, Weight: 136"
        ],
        "source_file": null
    },
    {
        "input": "What is happening within our body that causes the most stubborn acne when we were on our period? What is the difference between the acne we get while on our period versus off? What are the best ways (besides specific products) to treat acne specifically during that time?",
        "actual_output": "Hey! The stubborn acne around your period is mainly due to hormonal changes. Just before and during your period, your body’s progesterone and androgen levels rise, which can increase oil production in your skin. This excess oil can clog pores and lead to those stubborn pimples. Acne off your period is often less influenced by these hormone fluctuations and might be caused by other factors like stress, diet, or bacteria.\n\nTo help manage period-related acne without specific products, you can:\n- Keep your skin clean but avoid over-washing.\n- Maintain a balanced diet with plenty of water.\n- Manage stress through relaxation techniques.\n- Get enough sleep.\n- Avoid touching your face too much.\n\nIf your acne feels severe or painful, it’s a good idea to chat with a dermatologist or doctor for tailored advice.",
        "expected_output": "during menses body releases lots of hormones. these androgens cause stimulation of skin pilosebaceous ducts and lead to acne during the menstruation. these types of acne usually appear in chin area may be because of more concentrated pilosebaceous unit localized to this place. You are also still a teenager, so this is normal to have acne. You could try to maintain hormonal imbalance with the help of antiandrogen to reduce acne during your period, but please consult a dermatologist for further information",
        "retrieval_context": null,
        "context": [
            "Age: 16, BMI: 19.9485, CycleWithPeakorNot: 0, EndMenses: NA, Height: 66, IntercourseInFertileWindow: 1, LengthofCycle: 27, LengthofMenses: NA, MeanCycleLength: 27.17, MeanMensesLength: 5.33, NumberofDaysofIntercourse: 6, StartMenses: 2025-04-14, Today: 2025-04-14, UnusualBleeding: 1, Weight: 120"
        ],
        "source_file": null
    },
```

