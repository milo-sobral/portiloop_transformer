import argparse
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


def pretrain(wandb_group, wandb_project, wandb_exp_id, log_wandb=True, restore=False):

    # Initialize the config depending on the case
    if log_wandb:
        # Initialize WandB logging
        os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"  # TODO insert my own key
        wandb_run = wandb.init(
            project=wandb_project,
            group=wandb_group,
            id=wandb_exp_id,
            resume='allow',
            reinit=True,
            save_code=True)

        # Load the config
        if restore:
            config = wandb_run.config
            model_weights_filename = \
                f"model_{wandb_run.summary['batch'] // config['save_every'] * config['save_every']}.ckpt"
            restore_dict = torch.load(wandb_run.restore(model_weights_filename, run_path=wandb_run.path).name)

            logging.debug(f"Restoring model from {model_weights_filename}")
        else:
            config = initialize_config('test')
            if not validate_config(config):
                raise AttributeError("Issue with config.")
            wandb_run.config.update(config)
    else:
        # Load the config
        config = initialize_config('test')
        if not validate_config(config):
            raise AttributeError("Issue with config.")
        if restore:
            raise AttributeError("Cannot restore without WandB.")
        wandb_run = None
        
    logging.debug("Initializing pretraining...")

    # Basic initial setup
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.debug(f"Using device {config['device']}")

    # Set up all the dataset and model path
    home_path = pathlib.Path(__file__).parents[2].resolve()
    dataset_path = home_path / 'dataset' / 'MASS_preds'

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

    # Initialize the model
    model = TransformiloopPretrain(config)
    if restore:
        model_state_dict = restore_dict['model']
        model_state_dict = {k: v for k, v in model_state_dict.items() if "transformer.positional_encoder.pos_encoder.pe" != k}
        # Get the latest model weights filename, the highest multiple of save_every which is less than the last batch
        model.load_state_dict(model_state_dict)
    model.to(config['device'])

    # Initialize the pretraining optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=config["betas"]
    )
    if restore:
        optimizer.load_state_dict(restore_dict['optimizer'])

    # Initialize the Learning rate scheduler
    scheduler = WarmupTransformerLR(
        optimizer,
        config['warmup_steps'],
        config['lr'],
        config['lr_decay']
    )
    if restore:
        scheduler.load_state_dict(restore_dict['scheduler'])

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
                length=config['epoch_length'],
                last_batch=restore_dict['batch'] if restore else 0
            )
        except Exception as e:
            if wandb_run:
                wandb_run.finish()
            print(f"Pretraining stopped due to an exception during epoch {epoch}...")
            raise e


def finetune(wandb_group, wandb_project, wandb_exp_id, log_wandb=True, restore=False, pretrained_model=None):

    if pretrained_model is not None:
        if restore:
            raise AttributeError("Cannot restore and use a pretrained model. Either restore and already started finetuning run, or start a new finetuning run from a pretrained model.")
        api = wandb.Api()
        run = api.run(pretrained_model['run_path'])
        config = run.config
        model_dict = torch.load(\
                wandb.restore(pretrained_model['model_name'], run_path=pretrained_model['run_path']).name)
        # Initialize the model
        model = TransformiloopPretrain(config)
        model_state_dict = model_dict['model']
        model_state_dict = {k: v for k, v in model_state_dict.items() if "transformer.positional_encoder.pos_encoder.pe" != k}
        # Get the latest model weights filename, the highest multiple of save_every which is less than the last batch
        model.load_state_dict(model_state_dict)
        model.to(config['device'])
        
        # Initialize the model with the pretrained weights
        cnn_encoder, transformer = model.get_models()
        model = TransformiloopFinetune(config, cnn_encoder, transformer)
        
        if log_wandb:
            # Initialize WandB logging
            os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"  # TODO insert my own key
            wandb_run = wandb.init(
                project=wandb_project,
                group=wandb_group,
                id=wandb_exp_id,
                resume='allow',
                reinit=True,
                save_code=True)

        if not validate_config(config):
            raise AttributeError("Issue with config.")

        if log_wandb:
            wandb_run.config.update(config)
        else:
            wandb_run = None
    else:
        # Initialize the config depending on the case
        if log_wandb:
            # Initialize WandB logging
            os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"  # TODO insert my own key
            wandb_run = wandb.init(
                project=wandb_project,
                group=wandb_group,
                id=wandb_exp_id,
                resume='allow',
                reinit=True,
                save_code=True)

            # Load the config
            if restore:
                config = wandb_run.config
                model_weights_filename = \
                    f"model_{wandb_run.summary['batch'] // config['save_every'] * config['save_every']}.ckpt"
                model_dict = torch.load(wandb_run.restore(model_weights_filename, run_path=wandb_run.path).name)

                logging.debug(f"Restoring model from {model_weights_filename}")
            else:
                config = initialize_config('test')
                if not validate_config(config):
                    raise AttributeError("Issue with config.")
                wandb_run.config.update(config)
        else:
            # Load the config
            config = initialize_config('test')
            if not validate_config(config):
                raise AttributeError("Issue with config.")
            if restore:
                raise AttributeError("Cannot restore without WandB.")
            wandb_run = None

def run(config, wandb_group, wandb_project, save_model, unique_name, initial_validation=True):

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


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    # Mutually exclusive group of arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f', '--finetune', action='store_true', help='Finetune the model on the spindle dataset')
    group.add_argument('-p', '--pretrain', action='store_true', help='Pretrain the model on the MASS dataset')

    # Restore a run that exists
    parser.add_argument('-r', '--restore', action='store_true', default=False, help='Restore a run that exists. If that option is selected, make sure to provide a run name')

    # Configuration file
    parser.add_argument('--config', type=str, default=None, help='Path to the configuration file. If none is provided, the default configuration will be used')

    # WandB logging arguments
    parser.add_argument('-l', '--log_wandb', action='store_true', default=False, help='Log the run to WandB')
    parser.add_argument('-e', '--experiment_name', type=str, default=f"DEFAULT_EXPERIMENT", help='Name of the experiment. If none is provided, a unique experiment name will be generated.')
    parser.add_argument('-g', '--wandb_group', type=str, default='DEFAULT_GROUP', help='Name of the WandB group (default: DEFAULT_GROUP)')
    parser.add_argument('--wandb_project', type=str, default='Portiloop', help='Name of the WandB project (default: Portiloop)')

    args = parser.parse_args()
    
    # Check if the experiment name is valid
    if args.restore and args.experiment_name == "DEFAULT_EXPERIMENT":
        raise AttributeError("If you are restoring a run, you must provide a run name")

    # If the experiment name is not provided, generate a unique name
    if args.experiment_name == "DEFAULT_EXPERIMENT":
        args.experiment_name = f"Experiment_{time.time_ns()}"

    if args.finetune:
        finetune()
    elif args.pretrain:
        pretrain(args.wandb_group, args.wandb_project, args.experiment_name, log_wandb=args.log_wandb, restore=args.restore)
    else:
        raise AttributeError("Either pretrain or finetune must be selected")
    
    

    # run(config, 'experiment_clstoken_smallerlr', 'Milo-DEBUG', save_model, unique_name, pretrain, finetune_encoder)
    
