from typing import List, Optional
from pydantic import BaseModel


class ReasonScore(BaseModel):
    reason: str
    score: float


class BestTestCase(BaseModel):
    best_test_case_index: Optional[int] = None
    best_test_case_id: Optional[str] = None
    reason: str


class Steps(BaseModel):
    steps: List[str]
