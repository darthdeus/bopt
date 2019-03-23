import yaml
import os
import re
import psutil
import time
import pathlib
import datetime
import logging
import traceback

import numpy as np

from typing import List, Optional, Tuple

from bopt.basic_types import Hyperparameter, JobStatus, OptimizationFailed
from bopt.job_params import JobParams
from bopt.model_config import ModelConfig
from bopt.models.model import Model
from bopt.sample import Sample, SampleCollection
from bopt.models.parameters import ModelParameters
from bopt.models.random_search import RandomSearch
from bopt.runner.abstract import Job, Runner
from bopt.runner.runner_loader import RunnerLoader

from bopt.acquisition_functions.acquisition_functions import AcquisitionFunction

# TODO: set this at a proper global place
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("GP").setLevel(logging.WARNING)
# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger("matplotlib").setLevel(logging.INFO)



class NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, data):
        return True


class Experiment:
    kernel_names = ["rbf", "Mat32", "Mat52"]
    acquisition_fn_names = ["ei", "pi"]

    hyperparameters: List[Hyperparameter]
    runner: Runner
    samples: List[Sample]
    result_regex: str

    def __init__(self, hyperparameters: List[Hyperparameter], runner: Runner, result_regex: str):
        self.hyperparameters = hyperparameters
        self.runner = runner
        self.samples = []
        self.result_regex = result_regex

    def to_dict(self) -> dict:
        return {
            "hyperparameters": {h.name: h.to_dict() for h in self.hyperparameters},
            "samples": [s.to_dict() for s in self.samples],
            "runner": self.runner.to_dict(),
            "result_regex": self.result_regex
        }

    @staticmethod
    def from_dict(data: dict) -> "Experiment":
        hyperparameters = \
            [Hyperparameter.from_dict(key, data["hyperparameters"][key])
            for key in data["hyperparameters"].keys()]

        if data["samples"] and len(data["samples"]) > 0:
            samples = [Sample.from_dict(s, hyperparameters)
                    for s in data["samples"]]
        else:
            samples = []

        runner = RunnerLoader.from_dict(data["runner"])

        experiment = Experiment(hyperparameters, runner, data["result_regex"])
        experiment.samples = samples

        return experiment

    def collect_results(self) -> None:
        for sample in self.samples:
            if sample.result is None and sample.job and sample.job.is_finished():
                # Sine we're using `handle_cd` we always assume the working directory
                # is where meta.yml is.
                fname = os.path.join("output", f"job.o{sample.job.job_id}")

                if os.path.exists(fname):
                    sample.job.finished_at = datetime.datetime.now()

                    with open(fname, "r") as f:
                        contents = f.read().rstrip("\n")
                        found = False

                        for line in contents.split("\n"):
                            matches = re.match(self.result_regex, line)
                            if matches:
                                sample.result = float(matches.groups()[0])
                                found = True

                        if not found:
                            logging.error("Job {} seems to have failed, it finished running and its result cannot be parsed.".format(sample.job.job_id))
                else:
                    logging.error("Output file not found for job {} even though it finished. It will be considered as a failed job.".format(sample.job.job_id))

    def get_xy(self):
        samples = self.ok_samples()

        sample_col = SampleCollection(samples)
        X_sample, Y_sample = sample_col.to_xy()

        return X_sample, Y_sample

    def params_already_evaluated(self, job_params: JobParams) -> bool:
        return any([s.job.run_parameters.similar_to(job_params)
                    for s in self.samples if s.job])

    def suggest(self, model_config: ModelConfig) -> Tuple[JobParams, Model]:
        # TODO: overit, ze by to fungovalo i na ok+running a mean_pred
        if len(self.ok_samples()) == 0:
            logging.info("No existing samples found, overloading suggest with RandomSearch.")

            job_params, fitted_model = RandomSearch().predict_next(self.hyperparameters)
        else:
            from bopt.models.gpy_model import GPyModel

            X_sample, Y_sample = self.get_xy()

            try:
                job_params, fitted_model = GPyModel.predict_next(model_config, self.hyperparameters, X_sample, Y_sample)
            except OptimizationFailed as e:
                logging.error("Optimization failed, retrying with RandomSearch: {}".format(e))
                job_params, fitted_model = RandomSearch().predict_next(self.hyperparameters)

        return job_params, fitted_model

    def run_next(self, model_config: ModelConfig) -> Tuple[Job, Model, np.ndarray]:
        job_params, fitted_model = self.suggest(model_config)

        job, next_sample = self.manual_run(model_config, job_params,
                fitted_model.to_model_params())

        return job, fitted_model, next_sample.to_x()

    def manual_run(self, model_config: ModelConfig, job_params: JobParams,
            model_params: ModelParameters) -> Tuple[Job, Sample]:
        assert isinstance(job_params, JobParams)

        output_dir_path = pathlib.Path("output")
        output_dir_path.mkdir(parents=True, exist_ok=True)

        logging.info("Output set to {}, absolute path: {}".format(output_dir_path, output_dir_path.absolute()))

        job_params.validate()

        output_dir = str(output_dir_path)


        if self.params_already_evaluated(job_params):
            # TODO: opravit:
            #   - sample nemusi mit mu/sigma predikci
            #   - pokud uz byl vyhodnoceny, chci preskocit pousteni jobu a udelat "ManualSample"?
            raise NotImplementedError()

        job = self.runner.start(output_dir, job_params)

        X_sample, Y_sample = self.get_xy()

        if len(X_sample) > 0:
            from bopt.models.gpy_model import GPyModel

            if model_params.can_predict_mean():
                # Use the fitted model to predict mu/sigma.
                gpy_model = GPyModel.from_model_params(model_params, X_sample, Y_sample)
                model = gpy_model.model

            else:
                model = GPyModel.gpy_regression(model_config, X_sample, Y_sample)

            X_next = np.array([job_params.x])

            mu, var = model.predict(X_next)
            sigma = np.sqrt(var)

            assert mu.size == 1
            assert sigma.size == 1
        else:
            mu = 0.0
            sigma = 1.0 # TODO: lol :)

        next_sample = Sample(job, model_params, float(mu), float(sigma))
        self.samples.append(next_sample)

        self.serialize()
        logging.debug("Serialization done")

        return job, next_sample

    def run_single(self, model_config: ModelConfig) -> Job:
        job, fitted_model, x_next = self.run_next(model_config)

        # # TODO: nechci radsi JobParams?
        # logging.debug("Starting to plot")
        #
        # try:
        #     self.plot_current(fitted_model, meta_dir, x_next)
        # except ValueError as e:
        #     traceback_str = "".join(traceback.format_tb(e.__traceback__))
        #     logging.error("Plotting failed, error:\n{}\n\n".format(e, traceback_str))
        #
        # logging.debug("Plotting done")

        return job

    def serialize(self) -> None:
        dump = yaml.dump(self.to_dict(), default_flow_style=False, Dumper=NoAliasDumper)

        with open("meta.yml", "w") as f:
            f.write(dump)

    @staticmethod
    def deserialize() -> "Experiment":
        with open("meta.yml", "r") as f:
            contents = f.read()
            obj = yaml.load(contents, Loader=yaml.SafeLoader)

        experiment = Experiment.from_dict(obj)
        experiment.collect_results()
        experiment.serialize()

        return experiment

    def ok_samples(self) -> List[Sample]:
        return [s for s in self.samples if s.result]

    def num_running(self) -> int:
        return len([s for s in self.samples if s.status() == JobStatus.RUNNING])
