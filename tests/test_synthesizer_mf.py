from typing import Callable
import asyncio
import pytest
import time
import os
import json

from deepeval.synthesizer.chunking.context_generator import ContextGenerator
from deepeval.dataset import EvaluationDataset
from deepeval.dataset.golden import Golden
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import *
from deepeval.synthesizer import Evolution


#########################################################
### Generate Goldens From Scratch #######################
#########################################################

def test_generate_generate_goldens_from_scratch(synthesizer: Synthesizer):
    start_time = time.time()
    synthesizer.generate_goldens_from_scratch(
        num_goldens=5,
    )
    # print(goldens)
    print("=======Synthetic Goldens==========\n")
    print(synthesizer.synthetic_goldens)
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nTime taken: {duration} seconds")

    print(f"\nSynthesizer in pandas format:\n", synthesizer.to_pandas())

styling_config = StylingConfig(
    expected_output_format="string",
    input_format="Mimic the kind of queries or statements a female patient might share with a period and pregnancy tracking chatbot when seeking advice about her female health, pregnancy or period issues",
    task="The chatbot acts as a specialist in female gynecological issues (such as period and pregnancy complications) diagnosis and provides advice about their sex lifes.",
    scenario="Non-medical female customers of a period and pregnancy tracking app are describing their period or pregnancy symptoms to seek a diagnosis or seek sex advice.",
)

"""
DeepEval offers 7 types of evolutions: reasoning, multicontext, concretizing, constrained, comparative, hypothetical, and in-breadth evolutions.

- **Reasoning:** Evolves the input to require multi-step logical thinking.
- **Multicontext:** Ensures that all relevant information from the context is utilized.
- **Concretizing:** Makes abstract ideas more concrete and detailed.
- **Constrained:** Introduces a condition or restriction, testing the model's ability to operate within specific limits.
- **Comparative:** Requires a response that involves a comparison between options or contexts.
- **Hypothetical:** Forces the model to consider and respond to a hypothetical scenario.
- **In-breadth:** Broadens the input to touch on related or adjacent topics.
"""
evolution_config = EvolutionConfig(
    num_evolutions=2,
    evolutions={Evolution.COMPARATIVE: 0.2, Evolution.CONCRETIZING: 0.2, Evolution.HYPOTHETICAL: 0.4, Evolution.IN_BREADTH: 0.2},
)
synthesizer_async = Synthesizer(
    async_mode=True,
    styling_config=styling_config,
    evolution_config=evolution_config,
)
synthesizer = synthesizer_async
test_generate_generate_goldens_from_scratch(synthesizer)
