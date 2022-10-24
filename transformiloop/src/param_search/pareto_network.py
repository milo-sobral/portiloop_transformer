"""
Main script for PMBO

Possible arguments:
--server -> launches the server
--worker -> launches a worker
--meta -> launches the meta learner
--output_file -> name of the log file (string)
--ip_server -> ip of the server machine (string)

"""

from distutils.command.config import config
import logging
import socket
import time
from argparse import ArgumentParser
from copy import deepcopy
from threading import Lock, Thread

import torch
# from pyinstrument import Profiler
from requests import get

from transformiloop.src.param_search.pareto_network_server_utils import Server, RECV_TIMEOUT_META_FROM_SERVER, SOCKET_TIMEOUT_CONNECT_META, PORT_META, RECV_TIMEOUT_WORKER_FROM_SERVER, \
    PORT_WORKER, SOCKET_TIMEOUT_CONNECT_WORKER, ACK_TIMEOUT_WORKER_TO_SERVER, IP_SERVER, ACK_TIMEOUT_META_TO_SERVER, select_and_send_or_close_socket, poll_and_recv_or_close_socket, get_connected_socket, LOOP_SLEEP_TIME_META, LOOP_SLEEP_TIME_WORKER, LOOP_SLEEP_TIME, SEND_ALIVE
from transformiloop.src.param_search.pareto_search import LoggerWandbPareto, RUN_NAME, SurrogateModel, META_MODEL_DEVICE, train_surrogate, update_pareto, nb_parameters, MAX_NB_PARAMETERS, NB_SAMPLED_MODELS_PER_ITERATION, \
    exp_max_pareto_efficiency, exp_min_software_cost, load_network_files, dump_network_files, transform_config_dict_to_input, WANDB_PROJECT_PARETO, PARETO_ID, MAXIMIZE_F1_SCORE

from transformiloop.src.utils.train import run
from transformiloop.src.utils.configs import compare_configs, sample_config_dict


class Meta:
    def __init__(self, meta, timeout=-1, init_sample_size=1):
        self.meta = meta
        self.timeout = timeout
        self.init_sample_size = init_sample_size
        self.surrogate = None
        self.experiments = []
        self.launched = []
        self.best_exp = {}

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
            result = self.meta.pop(blocking=False)[0]
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

    def sample_experiment(self):
        """Samples ten experiments around our best so far and and returns the one with 
            the best expected return according to our surrogate model.
        
        Returns: 
            dict: The sampled dictionary.
        """
        return {}

    def learn_surrogate(self):
        """Learns a new surrogate model based on the dataset collected so far.
        """
        pass

    def load_experiments(self, file):
        """Loads previously run experiments from a file.

        Args:
            file (str): the file from which to load the model.
        """
        pass
    
# def meta(learner, timeout, init_sample_size):
#     logger = LoggerWandbPareto(RUN_NAME)
#     # finished_experiments, pareto_front = load_network_files()
#     # launched_experiments = []

#     # if finished_experiments is None:
#     #     logging.debug(f"no meta dataset found, starting new run")
#     #     finished_experiments = []  # list of dictionaries
#     #     pareto_front = []  # list of dictionaries, subset of finished_experiments
#     #     meta_model = SurrogateModel()
#     #     meta_model.to(META_MODEL_DEVICE)
#     # else:
#     #     logging.debug(f"existing meta dataset loaded")
#     #     logging.debug("training new surrogate model...")
#     #     meta_model = SurrogateModel()
#     #     meta_model.to(META_MODEL_DEVICE)
#     #     meta_model.train()
#     #     meta_model, meta_loss = train_surrogate(
#     #         meta_model, deepcopy(finished_experiments))
#     #     logging.debug(f"surrogate model loss: {meta_loss}")

#     # main meta-learning procedure:
#     finished_experiments = []
#     launched_experiments = []
#     num_exp = 0
#     prev_exp = {}

#     start_time = time.time()
#     while abs(time.time() - start_time) < timeout:
#         # Check which results were sent to us
#         # We do not want to block because we need to generate more experiments 
#         res = learner.receive_all(blocking=False)
#         # Train the surrogate model using the new point which arrived 
#         finished_experiments += res
#         if len(finished_experiments) > 0:
#             logging.debug("training new surrogate model...")

#             meta_model = SurrogateModel()
#             meta_model.to(META_MODEL_DEVICE)

#             meta_model.train()
#             meta_model, meta_loss = train_surrogate(
#                 meta_model, deepcopy(finished_experiments))

#             logging.debug(f"surrogate model loss: {meta_loss}")

#             logger.log(surrogate_loss=meta_loss,
#                         surprise=finished_experiments[-1]["surprise"], all_experiments=finished_experiments)
#             meta_model.eval()

#         num_exp = len(finished_experiments) + len(launched_experiments)
#         model_selected = False
#         exp = {}
#         exps = []

#         while not model_selected:
#             exp = {}

#             # sample model
#             config_dict, unrounded = sample_config_dict(exp_name=RUN_NAME + "_" + str(
#                 num_exp), prev_exp=prev_exp, all_exps=finished_experiments + launched_experiments + exps)

#             with torch.no_grad():
#                 input = transform_config_dict_to_input(config_dict)
#                 predicted_cost = meta_model(input).item()

#             exp["cost_software"] = predicted_cost
#             exp["config_dict"] = config_dict
#             exp["unrounded"] = unrounded
#             exp['cost_hardware'] = nb_parameters(config_dict)

#             exps.append(exp)

#             if len(exps) >= NB_SAMPLED_MODELS_PER_ITERATION:
#                 # select model
#                 model_selected = True
#                 if self._hardware_cost:  # minimize the Pareto tradeoff between hardware and software
#                     exp = exp_max_pareto_efficiency(exps, pareto_front, finished_experiments)
#                 else:  # minimize only the software cost (loss)
#                     exp = exp_min_software_cost(exps)

#         # logging.debug(f"config: {exp['config_dict']}")
#         # logging.debug(f"nb parameters: {exp['cost_hardware']}")
#         # logging.debug(f"predicted cost: {exp['cost_software']}")

#         # launched_experiments.append(exp)
#         # prev_exp = {}

def select_experiment(finished_experiments, launched_experiments, num_exp):
    pass

def worker(worker, timeout):
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
    if args.server:
        logging.debug("now running: server")
        # TODO: Launch a Relay
    elif args.worker:
        logging.debug("now running: worker")
        # TODO: Launch a Worker
    elif args.meta:
        logging.debug("now running: meta")
        # TODO: Launch a Meta Learner
    else:
        logging.debug("wrong argument")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--meta', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--ip_server', type=str, default=IP_SERVER)
    parser.add_argument('--dataset_path', type=str, default=None)

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
