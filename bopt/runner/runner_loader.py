from typing import Union, Type

from bopt.runner.abstract import Runner
from bopt.runner.local_runner import LocalRunner
from bopt.runner.sge_runner import SGERunner


class RunnerLoader:
    @staticmethod
    def from_dict(data: dict) -> Runner:
        runner_type = data["runner_type"]

        cls: Union[Type[LocalRunner], Type[SGERunner]]

        if runner_type == "local_runner":
            return LocalRunner(data["script_path"], data["arguments"])
        elif runner_type == "sge_runner":
            qsub_arguments = data.get("qsub_arguments", []) or []
            return SGERunner(data["script_path"], data["arguments"], qsub_arguments)
        else:
            raise NotImplementedError()

