import datetime

from typing import Optional, List

from bopt.basic_types import Hyperparameter
from bopt.runner.abstract import Job


class Sample:
    hyperparameters: List[Hyperparameter]
    other_params: dict

    started_at: datetime.datetime
    finished_at: datetime.datetime

    job: Job

    def y(self) -> Optional[float]:
        pass

