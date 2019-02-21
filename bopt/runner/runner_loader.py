from bopt.runner.abstract import Runner
from bopt.runner.local_runner import LocalRunner
from bopt.runner.sge_runner import SGERunner


class RunnerLoader:
    @staticmethod
    def from_dict(data: dict) -> Runner:
        runner_type = data["runner_type"]

        if runner_type == "local_runner":
            cls = LocalRunner
        elif runner_type == "sge_runner":
            cls = SGERunner
        else:
            raise NotImplemented()

        return cls(data["script_path"], data["arguments"])
