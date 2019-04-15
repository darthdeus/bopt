from typing import Union, Type

from bopt.runner.abstract import Runner
from bopt.runner.local_runner import LocalRunner
from bopt.runner.sge_runner import SGERunner


class RunnerLoader:
    @staticmethod
    def from_dict(data: dict) -> Runner:
        runner_type = data["runner_type"]

        cls: Union[Type[LocalRunner], Type[SGERunner]]
        manual_arg_fnames = data.get("manual_arg_fnames", []) or []

        if runner_type == "local_runner":
            return LocalRunner(data["script_path"], data["arguments"],
                    manual_arg_fnames)
        elif runner_type == "sge_runner":
            qsub_arguments = data.get("qsub_arguments", []) or []
            return SGERunner(data["script_path"], data["arguments"],
                    qsub_arguments, manual_arg_fnames)
        else:
            raise NotImplementedError()

