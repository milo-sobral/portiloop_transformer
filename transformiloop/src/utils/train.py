import copy
import logging
import os
import pprint
import time
from re import L
import pathlib
from torch.utils.data import DataLoader

import torch
from torch.utils.data.sampler import RandomSampler

import torch.optim as optim
import wandb
from torch.nn import BCEWithLogitsLoss
from torchinfo import summary
from transformiloop.src.data.pretraining import PretrainingDataset
from transformiloop.src.data.spindle_detection import get_dataloaders
from transformiloop.src.models.transformers import ClassificationTransformer, TransformiloopPretrain
from transformiloop.src.utils.configs import initialize_config, validate_config

from transformiloop.src.utils.train_utils import (finetune_epoch,
                                                  finetune_test_epoch,
                                                  WarmupTransformerLR, pretrain_epoch)


def pretrain(config, wandb_group, wandb_project, log_wandb=True):

    logging.debug("Initializing pretraining...")

    # Basic initial setup
    time_start = time.time()
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.debug(f"Using device {config['device']}")

    # Set up all the dataset and model path
    home_path = pathlib.Path(__file__).parents[2].resolve()
    dataset_path = home_path / 'dataset' / 'test_ds'
    models_path = home_path / 'models'
    models_path.mkdir(exist_ok=True)
    
    # Initialize the model
    model = TransformiloopPretrain(config).to(config['device'])

    logging.debug("Initializing dataset...")

    # Initialize the pretraining dataloader
    pre_dataset = PretrainingDataset(dataset_path, config)
    sampler = RandomSampler(data_source=pre_dataset, replacement=True)
    pretrain_dl = DataLoader(
        pre_dataset, 
        batch_size=config['batch_size'],
        sampler=sampler,
        pin_memory=True,
        drop_last=True
    )

    logging.debug("Done initializing dataloader")

    # Initialize the pretraining optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=config["betas"]
    )

    # Initialize the Learning rate scheduler
    scheduler = WarmupTransformerLR(
        optimizer,
        config['warmup_steps'],
        config['lr'],
        config['lr_decay']
    )

    # Set up a new WandB run
    exp_name = f"DEBUG_EXPERIMENT_2"

    if log_wandb:
        os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"  # TODO insert my own key
        wandb_run = wandb.init(
            project=wandb_project,
            id=exp_name,
            resume='allow',
            config=config,
            reinit=True,
            group=wandb_group,
            save_code=True)
    else:
        wandb_run = None

    for epoch in range(config['epochs']):
        try:
            logging.debug(f"Starting epoch {epoch}")
            pretrain_epoch(
                pretrain_dl, 
                config, 
                model, 
                optimizer,
                scheduler, 
                wandb_run,
                config['epoch_length']
            )
        except Exception as e:
            if wandb_run:
                wandb_run.finish()
            print(f"Pretraining stopped due to an exception during epoch {epoch}...")
            raise e


def run(config, wandb_group, wandb_project, save_model, unique_name, pretrain, finetune_encoder, initial_validation=True):

    time_start = time.time()

    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # config['d_model'] = config['window_size']

    dataset_path = pathlib.Path(__file__).parents[2].resolve() / 'dataset'
    # pretraining_data_path = dataset_path / 'pretraining_dataset.txt'
    # fi = os.path.join(DATASET_PATH, 'dataset_classification_full_big_250_matlab_standardized_envelope_pf.txt')

    # Initialize WandB logging
    experiment_name = f"{config['exp_name']}_{time.time_ns()}" if unique_name else config['exp_name']

    logging.debug(f"Config: {config}")
    logger = WandBLogger(wandb_group, config, wandb_project, experiment_name, dataset_path)

    # Load data
    train_dl, val_dl, _ = get_dataloaders(config, dataset_path)
    logging.debug(pprint.pprint(config))
    # pretraining_loader = data_generator(pretraining_data_path, config)

    # Load models
    classifier = ClassificationTransformer(config)
    print(summary(
        classifier,
        input_size=[
            (config['batch_size'], config['seq_len'], config['window_size']),
            (config['batch_size'], config['seq_len']-1)
        ],
        dtypes=[torch.float, torch.float, torch.float],
        depth=3,
    ))
    classifier.to(config['device'])

    # Initialize training objects
    config["loss_func"] = BCEWithLogitsLoss()

    config['classifier_optimizer'] = optim.Adam(
        classifier.parameters(),
        config["lr"],
        config["betas"])

    config["scheduler"] = WarmupTransformerLR(
        config["classifier_optimizer"],
        config['warmup_steps'],
        config['lr'],
        config['lr_decay']
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
    early_stopping_counter = 0

    # Start of Pretraining loop
    # min_loss = 10000.0
    # if pretrain:
    #     for epoch in range(config['epochs_pretrain']):
    #         loss, loss_t, loss_f, loss_c = pretrain_epoch(encoder, config['optimizer'], pretraining_loader, config, config['device'])
    #         logger.log({'loss': loss, 'loss_t': loss_t, 'loss_f': loss_f, 'loss_c':loss_c})
    #         if abs(min_loss - loss) <= config['es_delta']:
    #             break
    #     logging.debug("Done with pretraining...")
    
    # if not finetune_encoder:
    #     for param in encoder.parameters():
    #         param.requires_grad = False

    # Initial validation 
    if initial_validation:
        val_loss, val_acc, val_f1, val_rec, val_prec, val_cm = finetune_test_epoch(
                val_dl, config, classifier, config['device'])
        loggable_dict = {
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc,
            "Validation F1": val_f1,
            "Validation Recall": val_rec,
            "Validation Precision": val_prec,
            "Validation Confusion Matrix": val_cm
        }
        # logger.log(loggable_dict=loggable_dict)

    # Start of finetuning loop
    for epoch in range(config['epochs']):
        logging.debug(f"Starting epoch #{epoch}")
        print(f"Starting epoch #{epoch}")
        train_loss, train_acc, train_f1, train_rec, train_prec, train_cm, grads_flow = finetune_epoch(
            train_dl, config, config['device'], classifier, config['classifier_optimizer'], config['scheduler'])
        val_loss, val_acc, val_f1, val_rec, val_prec, val_cm = finetune_test_epoch(
            val_dl, config, classifier, config['device'])
        loggable_dict = {
            "Learning Rate": float(config['classifier_optimizer'].param_groups[0]['lr']),
            "Training Loss": train_loss,
            "Training Accuracy": train_acc,
            "Training F1": train_f1,
            "Training Recall": train_rec,
            "Training Precision": train_prec,
            "Training Confusion Matrix": train_cm,
            "Training Gradients Flow": grads_flow,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc,
            "Validation F1": val_f1,
            "Validation Recall": val_rec,
            "Validation Precision": val_prec,
            "Validation Confusion Matrix": val_cm
        }
        logger.log(loggable_dict=loggable_dict)
        config['classifier_optimizer'].step()

        def save_best_model():
            best_classifier = copy.deepcopy(classifier)

            if save_model:
                if save_model:
                    torch.save({
                        'epoch': epoch,
                        # 'encoder_state_dict': best_encoder.state_dict(),
                        # 'encoder_optimizer_state_dict': config['optimizer'].state_dict(),
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

        if early_stopping_counter > config['es_epochs'] or epoch > config['epochs']:
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

if __name__ == "__main__":
    
    config = initialize_config('test')
    if not validate_config(config):
        raise AttributeError("Issue with config.")
    save_model = False
    unique_name = True
    # pretrain = False
    finetune_encoder = True
    wandb_group = "Milo-DEBUG"
    wandb_project = "Portiloop"
    log_wandb = True
    # run(config, 'experiment_clstoken_smallerlr', 'Milo-DEBUG', save_model, unique_name, pretrain, finetune_encoder)
    pretrain(config, wandb_group, wandb_project, log_wandb)
