import math
import yaml
import os
import re
import psutil
import time
import pathlib
import datetime
import logging
import traceback
import tempfile

import numpy as np

from typing import List, Optional, Tuple

from bopt.basic_types import Hyperparameter, OptimizationFailed
from bopt.hyperparam_values import HyperparamValues
from bopt.gp_config import GPConfig
from bopt.models.model import Model
from bopt.sample import Sample, CollectFlag, SampleCollection
from bopt.models.parameters import ModelParameters
from bopt.models.random_search import RandomSearch
from bopt.runner.abstract import Job, Runner
from bopt.runner.runner_loader import RunnerLoader
from bopt.models.gpy_model import GPyModel


from bopt.acquisition_functions.acquisition_functions import AcquisitionFunction

# TODO: set this at a proper global place
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("GP").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
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

    gp_config: GPConfig

    def __init__(self, hyperparameters: List[Hyperparameter],
            runner: Runner, result_regex: str,
            gp_config: GPConfig) -> None:
        self.hyperparameters = hyperparameters
        self.runner = runner
        self.samples = []
        self.result_regex = result_regex
        self.gp_config = gp_config

    def to_dict(self) -> dict:
        return {
            "hyperparameters": {h.name: h.to_dict() for h in self.hyperparameters},
            "samples": [s.to_dict() for s in self.samples],
            "runner": self.runner.to_dict(),
            "result_regex": self.result_regex,
            "gp_config": self.gp_config
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

        experiment = Experiment(hyperparameters,
                runner,
                data["result_regex"],
                data["gp_config"])

        experiment.samples = samples

        return experiment

    def collect_results(self) -> None:
        # TODO: collect run time + check collected_at

        for sample in self.samples:
            if sample.collect_flag == CollectFlag.WAITING_FOR_SIMILAR:
                assert sample.result is None

                finished_similar_samples = self.get_finished_similar_samples(sample.hyperparam_values)

                if len(finished_similar_samples) > 0:
                    logging.info("Waiting for similar DONE, copying over results at {}".format(sample.hyperparam_values))

                    picked_similar = finished_similar_samples[0]

                    sample.result = picked_similar.result
                    sample.finished_at = datetime.datetime.now()
                    sample.collected_at = sample.finished_at
                    sample.collect_flag = CollectFlag.COLLECT_OK
                    sample.run_time = (sample.finished_at - sample.created_at).total_seconds()

            elif sample.collect_flag == CollectFlag.WAITING_FOR_JOB:
                assert sample.job
                assert sample.result is None

                if sample.job.is_finished():
                    # Sine we're using `handle_cd` we always assume the working
                    # directory is where meta.yml is.
                    fname = os.path.join("output", f"job.o{sample.job.job_id}")

                    if os.path.exists(fname):
                        with open(fname, "r") as f:
                            contents = f.read().rstrip("\n")
                            found = False

                            for line in contents.split("\n"):
                                bash_time_regex = r"real\t(\d+)m(\d+.\d+)s"

                                time_matches = re.match(bash_time_regex, line)

                                if time_matches:
                                    g = time_matches.groups()
                                    sample.run_time = int(g[0]) * 60 + float(g[1])
                                    sample.finished_at = sample.created_at + \
                                        datetime.timedelta(seconds=sample.run_time)

                                    logging.info("Collect parsed runtime of {}s"\
                                            .format(sample.run_time))

                                matches = re.match(self.result_regex, line)

                                if matches:
                                    sample.result = float(matches.groups()[0])
                                    sample.collected_at = datetime.datetime.now()
                                    sample.collect_flag = CollectFlag.COLLECT_OK
                                    found = True

                                    if not sample.run_time:
                                        logging.debug("No TIME parsed from the output, using `collected_at instead`.")
                                        sample.run_time = (sample.collected_at - sample.created_at).total_seconds()

                                    logging.info("Collect got result {}".format(sample.result))

                            if not found:
                                logging.error("Job {} seems to have failed, "
                                    "it finished running and its result cannot "
                                    "be parsed.".format(sample.job.job_id))

                                sample.collect_flag = CollectFlag.COLLECT_FAILED
                    else:
                        logging.error("Output file not found for job {} "
                            "even though it finished. It will be considered "
                            "as a failed job.".format(sample.job.job_id))

                        sample.collect_flag = CollectFlag.COLLECT_FAILED

    def samples_for_prediction(self) -> List[Sample]:
        return [s for s in self.samples if s.result or not s.model.sampled_from_random_search()]

    def predictive_samples_before(self, sample: Sample) -> List[Sample]:
        result = []

        for other in self.samples_for_prediction():
            other_date = other.finished_at or other.collected_at
            if not other_date:
                continue

            if other_date <= sample.created_at or sample == other:
                result.append(other)

        return result

    def get_xy(self):
        samples = self.samples_for_prediction()

        sample_col = SampleCollection(samples)
        X_sample, Y_sample = sample_col.to_xy()

        return X_sample, Y_sample

    def suggest(self) -> Tuple[HyperparamValues, Model]:
        job_params: HyperparamValues
        fitted_model: Model

        # TODO: overit, ze by to fungovalo i na ok+running a mean_pred
        if (len(self.samples_for_prediction()) < 2) or self.gp_config.random_search_only:
            logging.info("Sampling with random search.")

            job_params = RandomSearch.predict_next(self.hyperparameters)
            fitted_model = RandomSearch()
        else:
            X_sample, Y_sample = self.get_xy()

            try:
                job_params, fitted_model = GPyModel.predict_next(self.gp_config,
                        self.hyperparameters, X_sample, Y_sample)
            except OptimizationFailed as e:
                logging.error("Optimization failed, retrying with RandomSearch: "
                    "{}".format(e))

                job_params = RandomSearch.predict_next(self.hyperparameters)
                fitted_model = RandomSearch()

        return job_params, fitted_model

    def run_next(self, num_similar_retries: int = 5) -> Tuple[Model, Sample]:
        found_similar = True

        # This makes sure we try at least `num_similar_retries` times to re-run the job.
        while found_similar and num_similar_retries > 0:
            num_similar_retries -= 1

            job_params, fitted_model = self.suggest()

            next_sample, found_similar = self.manual_run(job_params,
                    fitted_model.to_model_params())

        return fitted_model, next_sample

    def get_similar_samples(self, hyperparam_values: HyperparamValues) \
            -> List[Sample]:
        return [s for s in self.samples
                if s.job and s.hyperparam_values.similar_to(hyperparam_values)]

    def get_finished_similar_samples(self, hyperparam_values: HyperparamValues) \
            -> List[Sample]:
        # Double filtering, but we don't care since there are only a few
        # samples anyway.
        return [s for s in self.get_similar_samples(hyperparam_values)
                if s.status() == CollectFlag.COLLECT_OK]

    def manual_run(self, hyperparam_values: HyperparamValues,
                         model_params: ModelParameters) -> Tuple[Sample, bool]:
        assert isinstance(hyperparam_values, HyperparamValues)

        output_dir_path = pathlib.Path("output")
        output_dir_path.mkdir(parents=True, exist_ok=True)

        logging.debug("Output set to: {}".format(output_dir_path,
            output_dir_path.absolute()))

        hyperparam_values.validate()

        output_dir = str(output_dir_path)

        similar_samples = self.get_similar_samples(hyperparam_values)
        found_similar = len(similar_samples) > 0

        if found_similar:
            finished_similar_samples = self.get_finished_similar_samples(hyperparam_values)

            if len(finished_similar_samples) > 0:
                warning_str = "Found finished similar sample, "
                warning_str += "creating MANUAL_SAMPLE with equal hyperparam values and result"
                warning_str += "... param values:\n{}\n{}".format(hyperparam_values, finished_similar_samples[0].hyperparam_values)

                logging.warning(warning_str)

                similar_sample = finished_similar_samples[0]
                assert similar_sample.result

                created_at = datetime.datetime.now()

                next_sample = Sample(None, model_params, hyperparam_values,
                        similar_sample.mu_pred, similar_sample.sigma_pred,
                        CollectFlag.COLLECT_OK, created_at)

                next_sample.collected_at = created_at
                next_sample.run_time = 0.0
                next_sample.result = similar_sample.result
                next_sample.comment = "created as similar of {}"\
                        .format(similar_sample)

            else:
                # TODO: opravit:
                #   - sample nemusi mit mu/sigma predikci
                #   - pokud uz byl vyhodnoceny, chci preskocit pousteni jobu a udelat "ManualSample"?
                similar_sample = similar_samples[0]

                next_sample = Sample(None, model_params, hyperparam_values,
                        similar_sample.mu_pred, similar_sample.sigma_pred,
                        CollectFlag.WAITING_FOR_SIMILAR,
                        datetime.datetime.now())

                next_sample.comment = "created as similar of {}"\
                        .format(similar_sample)

        else:
            manual_file_args = self.runner.fetch_and_shift_manual_file_args()
            job = self.runner.start(output_dir, hyperparam_values, manual_file_args)

            X_sample, Y_sample = self.get_xy()

            if len(X_sample) > 0:
                from bopt.models.gpy_model import GPyModel

                if model_params.can_predict_mean():
                    # Use the fitted model to predict mu/sigma.
                    gpy_model = GPyModel.from_model_params(self.gp_config,
                                                           model_params,
                                                           X_sample, Y_sample)

                    model = gpy_model.model

                else:
                    # TODO: gpy pouzito na 2 mistech?
                    model = GPyModel.gpy_regression(self.hyperparameters,
                            self.gp_config, X_sample, Y_sample)

                X_next = np.array([hyperparam_values.x])

                mu, var = model.predict(X_next)
                sigma = np.sqrt(var)

                mu = float(mu)
                sigma = float(sigma)

                assert not math.isnan(float(mu))
                assert not math.isnan(float(sigma))
            else:
                mu = None
                sigma = None

            next_sample = Sample(job, model_params, hyperparam_values,
                    mu, sigma, CollectFlag.WAITING_FOR_JOB,
                    datetime.datetime.now())

            next_sample.comment = " ".join(manual_file_args)

        self.samples.append(next_sample)

        self.serialize()
        logging.debug("Serialization done")

        return next_sample, found_similar

    def serialize(self) -> None:
        dump = yaml.dump(self.to_dict(), default_flow_style=False, Dumper=NoAliasDumper)

        temp_meta_fname = tempfile.mktemp(dir=".")

        with open(temp_meta_fname, "w") as f:
            f.write(dump)

        os.rename(temp_meta_fname, "meta.yml")

    @staticmethod
    def deserialize() -> "Experiment":
        with open("meta.yml", "r") as f:
            contents = f.read()
            obj = yaml.load(contents, Loader=yaml.Loader)

        experiment = Experiment.from_dict(obj)
        experiment.collect_results()
        experiment.serialize()

        return experiment
