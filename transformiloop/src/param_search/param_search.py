"""
Main script for PMBO

Possible arguments:
--server -> launches the server
--worker -> launches a worker
--meta -> launches the meta learner
--output_file -> name of the log file (string)
--ip_server -> ip of the server machine (string)

"""

import logging
import time
from argparse import ArgumentParser
from copy import deepcopy

import torch
# from pyinstrument import Profiler
from requests import get

from tlspyo import Relay, Endpoint

from transformiloop.src.param_search.learning_utils import LoggerWandbPareto, RUN_NAME, SurrogateModel, META_MODEL_DEVICE, train_surrogate, transform_config_dict_to_input, WANDB_PROJECT_PARETO, PARETO_ID, MAXIMIZE_F1_SCORE

from transformiloop.src.utils.train import run
from transformiloop.src.utils.configs import compare_configs, sample_config_dict

IP_SERVER = "" # TODO: Fill this with real IP
PASSWORD = "Transformiloop password"
SERVER_PORT = 3000
EXPLORATION_RATIO = 0.5

class Meta:
    def __init__(self, meta, run_name, timeout=-1, init_sample_size=1):
        self.meta = meta
        self.timeout = timeout
        self.init_sample_size = init_sample_size
        self.surrogate = None
        self.experiments = []
        self.launched = []
        self.best_exp = {}
        self.logger = LoggerWandbPareto(run_name)

    def run(self):
        """Runs the main behavior of the meta learner
        """
        # Load all the old experiments
        self.load_experiments()

        # Start by sampling the minimum necessary experiments
        for _ in range(self.init_sample_size):
            next_exp = self.sample_experiment()
            self.meta.produce(next_exp, "workers")
            self.launched.append(next_exp)

        # Launch the while loop
        start_time = time.time()
        while abs(time.time() - start_time) < self.timeout:
            # Block until we have a result from an experiment
            result = self.meta.pop(blocking=True)[0]
            # Once we do, retrain a new Surrogate model from scratch
            self.experiments.append(result)
            self.learn_surrogate()
            # Update the best experiment so far if necessary
            if result["best_f1_score"] > self.best_exp['best_f1_score']:
                self.best_exp = result
            # Once we have the model, sample X new experiments around our best so far and choose the one with the best expected return
            next_exp = self.sample_experiment()
            # Send that one to the workers group
            self.meta.produce(next_exp, "workers")
        self.meta.stop()

    def sample_experiment(self, num_to_sample=10):
        """Samples ten experiments around our best so far and and returns the one with 
            the best expected return according to our surrogate model.
        
        Returns: 
            dict: The sampled dictionary.
        """
        # Sample num_to_sample different experiments around the best so far
        pool_of_exps = []
        chance = random.uniform(0, 1)
        center = {} if chance > EXPLORATION_RATIO else self.best_exp
        for i in range(num_to_sample):
            pool_of_exps.append(sample_config_dict(f"experiment_{i}", center, self.experiments + pool_of_exps + self.launched))

        # Get the best expected result out of the ten samples and return it
        with torch.no_grad():
            best_exp = max(pool_of_exps, key=lambda elem: self.model(transform_config_dict_to_input(elem)))
        return best_exp

    def learn_surrogate(self):
        """Learns a new surrogate model based on the dataset collected so far.
        """
        logging.debug("training new surrogate model...")

        self.model = SurrogateModel()
        self.model.to(META_MODEL_DEVICE)
        self.model.train()
        self.model, meta_loss = train_surrogate(self.model, deepcopy(self.experiments))

        logging.debug(f"surrogate model loss: {meta_loss}")

        self.logger.log(surrogate_loss=meta_loss, surprise=self.experiments[-1]["surprise"], best_f1_score=self.best_exp["best_f1_score"])
        self.model.eval()

    def load_experiments(self, file):
        """Loads previously run experiments from a file.

        Args:
            file (str): the file from which to load the model.
        """
        raise NotImplementedError


def run_worker(worker, timeout):
    start_time = time.time()
    while abs(time.time() - start_time) < timeout:
        worker.notify("worker")
        exp = worker.pop(blocking=True)
        predicted_loss = exp['cost_software']

        logging.info(f"Launch run with predicted cost: {predicted_loss}")
        best_loss, best_f1_score, exp["best_epoch"] = run(
            exp["config_dict"], f"{WANDB_PROJECT_PARETO}_runs_{PARETO_ID}", 
            WANDB_PROJECT_PARETO, 
            save_model=False, 
            unique_name=True,
            pretrain=False,
            finetune_encoder=False)
        logging.debug("Run finished")
        exp["cost_software"] = 1 - \
            best_f1_score if MAXIMIZE_F1_SCORE else best_loss
        logging.info(f"run finished with actual cost: {exp['cost_software']} (predicted: {predicted_loss})")
        exp['surprise'] = exp["cost_software"] - predicted_loss
        
        worker.broadcast(exp, "meta")
    worker.stop()


def main(args, data_config=None):
    # Converting the timeout from hours to seconds
    timeout = args.timeout * 3600

    # Launching the necessary communication
    if args.server:
        logging.debug("now running: server")
        relay = Relay(
            port=SERVER_PORT,
            password=PASSWORD,
            accepted_groups=["meta", "worker"],
            local_com_port=3001,
            header_size=12
        )
        # Stay busy for as long as timeout
        start_time = time.time()
        while abs(time.time() - start_time) < timeout:
            time.sleep(60)
        relay.stop()
    elif args.worker:
        logging.debug("now running: worker")
        worker_ep = Endpoint(
            ip_server=IP_SERVER,
            port=SERVER_PORT,
            password=PASSWORD,
            groups="worker",
            local_com_port=3001,
            header_size=12
        )
        run_worker(worker_ep, timeout)
    elif args.meta:
        logging.debug("now running: meta")
        meta_ep = Endpoint(
            ip_server=IP_SERVER,
            port=SERVER_PORT,
            password=PASSWORD,
            groups="meta",
            local_com_port=3001,
            header_size=12
        )
        meta_trainer = Meta(meta_ep, timeout=timeout, init_sample_size=args.num_workers)
        meta_trainer.run()
    else:
        logging.debug("wrong argument")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--meta', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--ip_server', type=str, default = "127.0.0.1")
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--timeout', type=int, default=1)

    args = parser.parse_args()
    if args.output_file is not None:
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            filename=args.output_file, level=logging.DEBUG)
    else:
        logging.basicConfig(
            format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)

    # if args.worker:
    #     dataset_config = initialize_dataset_config(dataset_path=args.dataset_path)

    main(args) #, data_config=dataset_config)
