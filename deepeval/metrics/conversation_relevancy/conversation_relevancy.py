import asyncio
from typing import Optional, Union, Dict, List

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.conversation_relevancy.template import (
    ConversationRelevancyTemplate,
)
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
    get_turns_in_sliding_window,
    trimAndLoadJson,
    initialize_model,
    convert_turn_to_dict,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.conversation_relevancy.schema import *


class ConversationRelevancyMetric(BaseConversationalMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        window_size: int = 10,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.window_size = window_size

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ):
        check_conversational_test_case_params(test_case, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                    )
                )
            else:
                turns_windows: List[List[Turn]] = [
                    window
                    for window in get_turns_in_sliding_window(
                        test_case.turns, self.window_size
                    )
                ]

                self.verdicts = [
                    self._generate_verdict(window) for window in turns_windows
                ]

                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Turns Sliding Windows (size={self.window_size}):\n{prettify_list(turns_windows)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
            return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_conversational_test_case_params(test_case, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            turns_windows: List[List[Turn]] = [
                window
                for window in get_turns_in_sliding_window(
                    test_case.turns, self.window_size
                )
            ]

            self.verdicts = await asyncio.gather(
                *[self._a_generate_verdict(window) for window in turns_windows]
            )

            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Turns Sliding Windows (size={self.window_size}):\n{prettify_list(turns_windows)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        irrelevancies: List[Dict[str, str]] = []
        for index, verdict in enumerate(self.verdicts):
            if verdict.verdict.strip().lower() == "no":
                irrelevancies.append(
                    {"message number": f"{index+1}", "reason": verdict.reason}
                )

        prompt = ConversationRelevancyTemplate.generate_reason(
            score=self.score, irrelevancies=irrelevancies
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=ConversationRelevancyScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: ConversationRelevancyScoreReason = (
                    await self.model.a_generate(
                        prompt, schema=ConversationRelevancyScoreReason
                    )
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self) -> str:
        irrelevancies: List[Dict[str, str]] = []
        for index, verdict in enumerate(self.verdicts):
            if verdict.verdict.strip().lower() == "no":
                irrelevancies.append(
                    {"message number": f"{index+1}", "reason": verdict.reason}
                )

        prompt = ConversationRelevancyTemplate.generate_reason(
            score=self.score, irrelevancies=irrelevancies
        )
        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=ConversationRelevancyScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: ConversationRelevancyScoreReason = self.model.generate(
                    prompt, schema=ConversationRelevancyScoreReason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_verdict(
        self, turns_sliding_window: List[Turn]
    ) -> ConversationRelevancyVerdict:
        prompt = ConversationRelevancyTemplate.generate_verdicts(
            sliding_window=[
                convert_turn_to_dict(turn) for turn in turns_sliding_window
            ]
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=ConversationRelevancyVerdict
            )
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: ConversationRelevancyVerdict = await self.model.a_generate(
                    prompt, schema=ConversationRelevancyVerdict
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return ConversationRelevancyVerdict(**data)

    def _generate_verdict(
        self, turns_sliding_window: List[Turn]
    ) -> ConversationRelevancyVerdict:
        prompt = ConversationRelevancyTemplate.generate_verdicts(
            sliding_window=[
                convert_turn_to_dict(turn) for turn in turns_sliding_window
            ]
        )
        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=ConversationRelevancyVerdict
            )
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: ConversationRelevancyVerdict = self.model.generate(
                    prompt, schema=ConversationRelevancyVerdict
                )
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return ConversationRelevancyVerdict(**data)

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        relevant_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                relevant_count += 1

        score = relevant_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Conversation Relevancy"
