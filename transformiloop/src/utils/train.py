import logging
import os
import wandb
import torch
from torchinfo import summary
import pprint
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim


from transformiloop.src.utils.train_utils import pretrain_epoch, finetune_epoch, finetune_test_epoch
from transformiloop.src.utils.configs import get_default_config
from transformiloop.src.data.spindle_detect_data import get_dataloaders
from transformiloop.src.data.pretraining_data import data_generator
from transformiloop.src.models.model_factory import get_encoder_classifier_TFC

def run(config, wandb_group, wandb_project, save_model, unique_name):

    # Initialize WandB logging
    logging.debug(f"Config: {config}")
    logger = WandBLogger(wandb_group, config, wandb_project, unique_name)

    # Load models
    classifier, encoder = get_encoder_classifier_TFC(config)
    logging.debug(summary(
        classifier,
        input_size=[
            (config['batch_size'], config['MODA_data_config']['seq_len'], config['encoder_config']['d_model'])
        ],
        dtypes=[torch.float, torch.float, torch.bool],
        depth=3,
    ))
    logging.debug(summary(
        encoder,
        input_size=[
            (config['batch_size'], 1, config['seq_len']),
            (config['batch_size'], 1, config['seq_len'])
        ],
        dtypes=[torch.float, torch.float, torch.bool],
        depth=3,
    ))

    # Load data
    train_dl, val_dl, test_dl = get_dataloaders(config)
    logging.debug(pprint.pprint(config))

    # Initialize training objects
    config["loss_func"] = BCEWithLogitsLoss()

    config["optimizer"] = optim.Adam(
        encoder.parameters(),
        config['training']["lr"],
        config['training']["betas"])

    config['classifier_optimizer'] = optim.Adam(
        classifier.parameters(),
        config['training']["lr"],
        config['training']["betas"])


    config["scheduler"] = optim.lr_scheduler.StepLR(
        config["optimizer"], 
        step_size=1000,
        gamma=0.1
    )

    # Start of training loop
    for epoch in range(config['epochs']):
        loss, acc, f1, recall, precision, cm_train = finetune_epoch(encoder, config['optimizer'], train_dl, config, config['device'], classifier, config['classifier_optimizer'], 99)
        acc_test, f1_test, rec_test, prec_test, cm_test = finetune_test_epoch(encoder, val_dl, config, classifier, config['device'], 39)
        # TODO log here
    # Send results

    del(logger)


class WandBLogger:
    def __init__(self, group_name, config, project_name, experiment_name):
        self.best_model = None
        self.experiment_name = experiment_name
        self.config = config
        os.environ['WANDB_API_KEY'] = "" # TODO Insert my own key
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


    def update_best_model(self, model):
        self.best_model = model
        self.wandb_run.save(os.path.join(
            self.config['subjects_path'], 
            self.experiment_name + "_encoder"), 
            policy="live", 
            base_path=self.self.config['subjects_path'])
        self.wandb_run.save(os.path.join(
            self.config['subjects_path'], 
            self.experiment_name + "_classifier"), 
            policy="live", 
            base_path=self.self.config['subjects_path'])
        
    def __del__(self):
        self.wandb_run.finish()

    def restore(self):
        self.wandb_run.restore(self.experiment_name, root=self.config['subjects_path'])
        