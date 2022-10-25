"""
Pareto-optimal hyperparameter search (meta-learning)
"""
import logging
import os
import pickle as pkl
# all imports
import random
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# from transformiloop.src.param_search.pareto_network import MAXIMIZE_F1_SCORE
from transformiloop.src.utils.configs import get_default_config, SAMPLEABLE_DICT
from transformiloop.src.utils.configs import sample_config_dict
from transformiloop.src.utils.train import run

import wandb


# all constants (no hyperparameters here!)
# from portiloop_software.portiloop_python.ANN.training_experiment import PortiloopNetwork, initialize_exp_config, run, initialize_dataset_config
# from portiloop_software.portiloop_python.Utils.utils import EPSILON_EXP_NOISE, MAX_NB_PARAMETERS, MIN_NB_PARAMETERS, sample_config_dict, MAXIMIZE_F1_SCORE
MAX_NB_PARAMETERS = 2000000
EPSILON_EXP_NOISE = 0.1
MIN_NB_PARAMETERS = 100000

MAXIMIZE_F1_SCORE = True


THRESHOLD = 0.2
WANDB_PROJECT_PARETO = "Milo-SEARCH"

path_dataset = Path(__file__).absolute().parent.parent.parent / 'dataset'
path_pareto = Path(__file__).absolute().parent.parent.parent / 'pareto'

# path = "/content/drive/MyDrive/Data/MASS/"
# path_dataset = Path(path)
# path_pareto = Path("/content/drive/MyDrive/Data/pareto_results/")

MAX_META_ITERATIONS = 1000  # maximum number of experiments

# MAX_LOSS = 0.1  # to normalize distances

META_MODEL_DEVICE = "cpu"  # the surrogate model will be trained on this device

PARETO_ID = "06"
RUN_NAME = f"pareto_search_{PARETO_ID}"

# number of models sampled per iteration, only the best predicted one is selected
NB_SAMPLED_MODELS_PER_ITERATION = 200

# default number of meta-epochs before entering meta train/val training regime
DEFAULT_META_EPOCHS = 100
# minimum number of experiments in the dataset before using a validation set
START_META_TRAIN_VAL_AFTER = 100
META_TRAIN_VAL_RATIO = 0.8  # portion of experiments in meta training sets
# surrogate training will stop after this number of meta-training epochs if the model doesn't converge
MAX_META_EPOCHS = 500
# meta early stopping after this number of unsuccessful meta epochs
META_EARLY_STOPPING = 30


class MetaDataset(Dataset):
    def __init__(self, finished_runs, start, end):
        size = len(finished_runs)
        self.data = finished_runs[int(start * size):int(end * size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert 0 <= idx <= len(
            self), f"Index out of range ({idx}/{len(self)})."
        config_dict = self.data[idx]["config_dict"]
        x = transform_config_dict_to_input(config_dict)
        label = torch.tensor(self.data[idx]["cost_software"])
        return x, label


class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        nb_features = len(SAMPLEABLE_DICT.keys())
        coeff = 20
        self.fc1 = nn.Linear(in_features=nb_features,  # nb hyperparameters
                             out_features=nb_features * coeff)  # in SMBO paper : 25 * hyperparameters... Seems huge

        self.d1 = nn.Dropout(0)

        self.fc2 = nn.Linear(in_features=nb_features * coeff,
                             out_features=nb_features * coeff)

        self.d2 = nn.Dropout(0)

        self.fc3 = nn.Linear(in_features=nb_features * coeff,
                             out_features=1)

    def to(self, device):
        super(SurrogateModel, self).to(device)
        self.device = device

    def forward(self, x):
        x_tensor = x.to(self.device)
        x_tensor = F.relu(self.d1(self.fc1(x_tensor)))
        x_tensor = F.relu(self.d2(self.fc2(x_tensor)))

        # x_tensor = F.relu(self.fc1(x_tensor))
        # x_tensor = F.relu(self.fc2(x_tensor))

        x_tensor = self.fc3(x_tensor)

        return x_tensor


def transform_config_dict_to_input(config_dict):
    return  torch.tensor([float(config_dict[key]) for key in config_dict.keys() if key in SAMPLEABLE_DICT.keys()])


def train_surrogate(net, all_experiments):
    optimizer = torch.optim.SGD(net.parameters(), \
        lr=0.05, momentum=0, dampening=0, weight_decay=0.01, nesterov=False)
    criterion = nn.MSELoss()
    best_val_loss = np.inf
    best_model = None
    early_stopping_counter = 0
    random.shuffle(all_experiments)
    max_epoch = MAX_META_EPOCHS if len(
        all_experiments) > START_META_TRAIN_VAL_AFTER else len(all_experiments)

    for epoch in range(max_epoch):
        if len(all_experiments) > START_META_TRAIN_VAL_AFTER:
            train_dataset = MetaDataset(
                all_experiments, start=0, end=META_TRAIN_VAL_RATIO)
            validation_dataset = MetaDataset(
                all_experiments, start=META_TRAIN_VAL_RATIO, end=1)
            train_loader = DataLoader(
                train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
            validation_loader = DataLoader(
                validation_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
        else:
            train_dataset = MetaDataset(all_experiments, start=0, end=1)
            train_loader = DataLoader(
                train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
        losses = []

        net.train()
        for batch_data in train_loader:
            batch_samples, batch_labels = batch_data
            batch_samples = batch_samples.to(device=META_MODEL_DEVICE).float()
            batch_labels = batch_labels.to(device=META_MODEL_DEVICE).float()

            optimizer.zero_grad()
            output = net(batch_samples)
            output = output.view(-1)
            loss = criterion(output, batch_labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()

        mean_loss = np.mean(losses)
        # logging.debug(f"DEBUG: epoch {epoch} mean_loss_training = {mean_loss}")

        if len(all_experiments) > START_META_TRAIN_VAL_AFTER:
            net.eval()
            losses = []
            with torch.no_grad():
                for batch_data in validation_loader:
                    batch_samples, batch_labels = batch_data
                    batch_samples = batch_samples.to(
                        device=META_MODEL_DEVICE).float()
                    batch_labels = batch_labels.to(
                        device=META_MODEL_DEVICE).float()

                    output = net(batch_samples)
                    output = output.view(-1)
                    loss = criterion(output, batch_labels)
                    losses.append(loss.item())

                mean_loss_validation = np.mean(losses)
                # logging.debug(f"DEBUG: mean_loss_validation = {mean_loss_validation}")
                if mean_loss_validation < best_val_loss:
                    best_val_loss = mean_loss_validation
                    early_stopping_counter = 0
                    best_model = deepcopy(net)
                else:
                    early_stopping_counter += 1
                # early stopping:
                if early_stopping_counter >= META_EARLY_STOPPING:
                    net = best_model
                    mean_loss = best_val_loss
                    logging.debug(
                        f"DEBUG: meta training converged at epoch:{epoch} (-{META_EARLY_STOPPING})")
                    break
                elif epoch == MAX_META_EPOCHS - 1:
                    logging.debug(
                        f"DEBUG: meta training did not converge after epoch:{epoch}")
                    break
    net.eval()
    return net, mean_loss


def dump_files(all_experiments, pareto_front):
    """
    exports pickled files to path_pareto
    """
    path_current_all = path_pareto / (RUN_NAME + "_all.pkl")
    path_current_pareto = path_pareto / (RUN_NAME + "_pareto.pkl")
    with open(path_current_all, "wb") as f:
        pkl.dump(all_experiments, f)
    with open(path_current_pareto, "wb") as f:
        pkl.dump(pareto_front, f)


def load_files():
    """
    loads pickled files from path_pareto
    returns None, None if not found
    else returns all_experiments, pareto_front
    """
    path_current_all = path_pareto / (RUN_NAME + "_all.pkl")
    path_current_pareto = path_pareto / (RUN_NAME + "_pareto.pkl")
    if not path_current_all.exists() or not path_current_pareto.exists():
        return None, None
    with open(path_current_all, "rb") as f:
        all_experiments = pkl.load(f)
    with open(path_current_pareto, "rb") as f:
        pareto_front = pkl.load(f)
    return all_experiments, pareto_front


def dump_network_files(finished_experiments, pareto_front):
    """
    exports pickled files to path_pareto
    """
    path_current_finished = path_pareto / (RUN_NAME + "_finished.pkl")
    path_current_pareto = path_pareto / (RUN_NAME + "_pareto.pkl")
    #   path_current_launched = path_pareto / (RUN_NAME + "_launched.pkl")
    with open(path_current_finished, "wb") as f:
        pkl.dump(finished_experiments, f)
    #  with open(path_current_launched, "wb") as f:
    #     pkl.dump(launched_experiments, f)
    with open(path_current_pareto, "wb") as f:
        pkl.dump(pareto_front, f)


def load_network_files():
    """
    loads pickled files from path_pareto
    returns None, None if not found
    else returns all_experiments, pareto_front
    """
    path_current_finished = path_pareto / (RUN_NAME + "_finished.pkl")
    path_current_pareto = path_pareto / (RUN_NAME + "_pareto.pkl")
    #  path_current_launched = path_pareto / (RUN_NAME + "_launched.pkl")
    if not path_current_finished.exists() or not path_current_pareto.exists():
        return None, None
    with open(path_current_finished, "rb") as f:
        finished_experiments = pkl.load(f)
    with open(path_current_pareto, "rb") as f:
        pareto_front = pkl.load(f)
    #  with open(path_current_launched, "rb") as f:
    #     launched_experiments = pkl.load(f)
    return finished_experiments, pareto_front


class LoggerWandbPareto:
    def __init__(self, run_name):
        self.run_name = run_name
        os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"
        self.wandb_run = wandb.init(
            project=WANDB_PROJECT_PARETO, entity="portiloop", id=run_name, resume="allow", reinit=True)

    def log(self,
            surrogate_loss,
            surprise,
            best_f1_score
            ):
        wandb.log({
            "surrogate_loss": surrogate_loss,
            "surprise": surprise,
            "best_f1_score": best_f1_score
        })

    def __del__(self):
        self.wandb_run.finish()

