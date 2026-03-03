from dataclasses import dataclass
from typing import List

from pydantic import BaseModel


@dataclass
class Interval:
    label: str
    start: int
    end: int  # inclusive

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class CapturesClassification(BaseModel):
    predictions: List[Interval]
