__all__ = [
    "CycleCorrectionSettings",
]
from dataclasses import dataclass
from typing import Optional

@dataclass
class CycleCorrectionSettings:
    num_cycles: Optional[int]  # how often
    test_input_value: Optional[float]  # what to feed to check if working
    relative_deviation: Optional[float]  # acceptable deviation
    ideal_programming: bool # whether to assume perfect programming, requires every other value to be None

    def __post_init__(self):
        if self.ideal_programming is True:
            # ideal programming does not use these values
            assert self.num_cycles is None
            assert self.test_input_value is None
            assert self.relative_deviation is None