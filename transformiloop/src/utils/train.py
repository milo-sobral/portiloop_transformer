import copy
import logging
import os
import pprint
import time
from re import L
import pathlib

import torch
import torch.optim as optim
import wandb
from torch.nn import BCEWithLogitsLoss
from torchinfo import summary
from transformiloop.src.data.pretraining_data import data_generator
from transformiloop.src.data.spindle_detect_data import get_dataloaders
from transformiloop.src.models.model_factory import get_encoder_classifier_TFC
from transformiloop.src.utils.configs import get_default_config
from transformiloop.src.utils.train_utils import (finetune_epoch,
                                                  finetune_test_epoch,
                                                  pretrain_epoch)


def run(config, wandb_group, wandb_project, save_model, unique_name):

    time_start = time.time()
    dataset_path = pathlib.Path(__file__).parents[2].resolve() / 'dataset'
    # fi = os.path.join(DATASET_PATH, 'dataset_classification_full_big_250_matlab_standardized_envelope_pf.txt')

    # Initialize WandB logging
    experiment_name = f"{config['exp_name']}_{time.time_ns()}" if unique_name else config['exp_name']

    logging.debug(f"Config: {config}")
    logger = WandBLogger(wandb_group, config, wandb_project, experiment_name, dataset_path)

    # Load models
    classifier, encoder = get_encoder_classifier_TFC(config)
    logging.debug(summary(
        classifier,
        input_size=[
            (config['batch_size'], config['seq_len'], config['d_model'])
        ],
        dtypes=[torch.float, torch.float, torch.bool],
        depth=3,
    ))
    logging.debug(summary(
        encoder,
        input_size=[
            (config['batch_size'], 1, config['window_size']),
            (config['batch_size'], 1, config['window_size'])
        ],
        dtypes=[torch.float, torch.float, torch.bool],
        depth=3,
    ))

    # Load data
    train_dl, val_dl, test_dl = get_dataloaders(config, dataset_path)
    logging.debug(pprint.pprint(config))

    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize training objects
    config["loss_func"] = BCEWithLogitsLoss()

    config["optimizer"] = optim.Adam(
        encoder.parameters(),
        config["lr"],
        config["betas"])

    config['classifier_optimizer'] = optim.Adam(
        classifier.parameters(),
        config["lr"],
        config["betas"])

    config["scheduler"] = optim.lr_scheduler.StepLR(
        config["optimizer"],
        step_size=1000,
        gamma=0.1
    )
    loss_es = None
    best_epoch = 0
    updated_model = False

    best_loss_early_stopping = 1
    best_epoch_early_stopping = 0
    best_model_precision_validation = 0
    best_model_f1_score_validation = 0
    best_model_recall_validation = 0
    best_model_loss_validation = 1

    # Start of training loop
    for epoch in range(config['epochs']):
        train_loss, train_acc, train_f1, train_rec, train_prec, train_cm = finetune_epoch(
            encoder, config['optimizer'], train_dl, config, config['device'], classifier, config['classifier_optimizer'], 99)
        val_loss, val_acc, val_f1, val_rec, val_prec, val_cm = finetune_test_epoch(
            encoder, val_dl, config, classifier, config['device'], 39)
        loggable_dict = {
            "Training Loss": train_loss,
            "Training Accuracy": train_acc,
            "Training F1": train_f1,
            "Training Recall": train_rec,
            "Training Precision": train_prec,
            "Training Confusion Matrix": train_cm,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc,
            "Validation F1": val_f1,
            "Validation Recall": val_rec,
            "Validation Precision": val_prec,
            "Validation Confusion Matrix": val_cm
        }
        logger.log(loggable_dict=loggable_dict)

        def save_best_model():
            best_encoder = copy.deepcopy(encoder)
            best_classifier = copy.deepcopy(classifier)

            if save_model:
                if save_model:
                    torch.save({
                        'epoch': epoch,
                        'encoder_state_dict': best_encoder.state_dict(),
                        'encoder_optimizer_state_dict': config['optimizer'].state_dict(),
                        'classifier_state_dict': best_classifier.state_dict(),
                        'classifier_optimizer_state_dict': config['classifier_optimizer'].state_dict(),
                        'recall_validation': best_model_recall_validation,
                        'precision_validation': best_model_precision_validation,
                        'loss_validation': best_model_loss_validation,
                        'f1_score_validation': best_model_f1_score_validation,
                        'accuracy_validation': best_model_accuracy
                    }, dataset_path / experiment_name, _use_new_zipfile_serialization=False)

        # Check if we have the best epoch
        if val_f1 > best_model_f1_score_validation:
            best_epoch = epoch
            best_model_f1_score_validation = val_f1
            best_model_precision_validation = val_prec
            best_model_recall_validation = val_rec
            best_model_loss_validation = val_loss
            best_model_accuracy = val_acc
            logger.update_summary(
                best_epoch,
                best_model_f1_score_validation,
                best_model_precision_validation,
                best_model_recall_validation,
                best_model_loss_validation,
                best_model_accuracy
            )
            save_best_model()
            logger.update_best_model()

        def update_es_loss(validation_loss, es_smoothing_factor):
            if loss_es is None:
                return validation_loss
            else:
                return validation_loss * es_smoothing_factor + loss_es * (1.0 - es_smoothing_factor)

        loss_es = update_es_loss(val_loss, 0.1)
        if loss_es < best_loss_early_stopping:
            best_loss_early_stopping = loss_es
            early_stopping_counter = 0
            best_epoch_early_stopping = epoch
        else:
            early_stopping_counter += 1

        if early_stopping_counter > config['es_epochs'] or epoch > config['epochs'] or time.time() - time_start > config['max_duration']:
            logging.debug("Early Stopping")
            break

    del (logger)
    return best_model_loss_validation, best_model_f1_score_validation, best_epoch_early_stopping


class WandBLogger:
    def __init__(self, group_name, config, project_name, experiment_name, dataset_path):
        self.best_model = None
        self.experiment_name = experiment_name
        self.config = config
        self.dataset_path = dataset_path
        os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"  # TODO insert my own key
        self.wandb_run = wandb.init(
            project=project_name,
            id=experiment_name,
            resume='allow',
            config=config,
            reinit=True,
            group=group_name,
            save_code=True)

    def log(self, loggable_dict):
        self.wandb_run.log(loggable_dict)

    def update_summary(
        self,
        best_epoch,
        best_f1_score,
        best_precision,
        best_recall,
        best_loss,
        best_accuracy
    ):
        self.wandb_run.summary['best_epoch'] = best_epoch
        self.wandb_run.summary['best_f1_score'] = best_f1_score
        self.wandb_run.summary['best_precision'] = best_precision
        self.wandb_run.summary['best_recall'] = best_recall
        self.wandb_run.summary['best_loss'] = best_loss
        self.wandb_run.summary['best_accuracy'] = best_accuracy

    def update_best_model(self):
        self.wandb_run.save(os.path.join(
            self.dataset_path,
            self.experiment_name),
            policy="live",
            base_path=self.dataset_path)

    def __del__(self):
        self.wandb_run.finish()

    def restore(self):
        self.wandb_run.restore(self.experiment_name,
                               root=self.dataset_path)
