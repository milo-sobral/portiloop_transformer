import torch
import numpy as np
import torch.nn as nn
import os
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging
import wandb
from torch.optim.lr_scheduler import _LRScheduler
from copy import deepcopy
from sklearn.metrics import classification_report

from transformiloop.src.data.sleep_stage import SleepStageDataset


def save_model(model, optimizer, scheduler, batch_idx, exp_name):
    """
    Saves the model and config to a temporary directory and then uploads it to wandb.
    """

    logging.debug(f"Saving model at batch {batch_idx}")

    # Create a temporary directory to save the model and config from path where script is running
    temp_path = os.path.join(f"transformiloop/models/", exp_name)

    # Create the temporary directory
    if not os.path.exists(temp_path):
        os.system('mkdir -p ' + temp_path)
        
    # Save the model and config to the temporary directory
    model_state_dict = model.state_dict()
    model_state_dict = {k: v for k, v in model_state_dict.items() if "transformer.positional_encoder.pos_encoder.pe" != k}
    to_save = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'batch': batch_idx,
    }
    torch.save(to_save, os.path.join(temp_path, f"model_{batch_idx}.ckpt"))

    # Upload the model and config to wandb
    wandb.save(temp_path + '/*', base_path=temp_path)


def seq_rec_loss(predictions, expected, mask, loss):
    '''
    Compute the loss for sequence reconstruction.
    Predictions: (batch_size, seq_len, reconstruction_dim)
    Expected: (batch_size, seq_len, reconstruction_dim)
    Mask: (batch_size, seq_len)
    loss: loss function to use
    '''
    mask = torch.where(mask==0, mask, 1).unsqueeze(-1)
    mask = mask.expand(mask.size(0), mask.size(1), predictions.size(-1))
    predictions = predictions * mask
    expected = expected * mask
    return loss(predictions, expected)
    

def pretrain_epoch(dataloader, config, model, optim, scheduler, wandblogger, length=-1, last_batch=0):
    model.train()

    mse_loss = nn.MSELoss()
    loss_fns = {
        'gender': nn.BCEWithLogitsLoss(),
        'age': mse_loss,
        'seq_rec': (lambda pred, exp, mask: seq_rec_loss(pred, exp, mask, mse_loss)),

    }

    for batch_idx, batch in enumerate(dataloader):
        optim.zero_grad()

        # Skip batches that have already been trained
        batch_idx += last_batch

        # Stop after a certain number of batches
        if length > 0 and batch_idx > length:
            break

        if batch_idx % config['log_every'] == 0:
            logging.debug(f"Training batch {batch_idx}")

        # run through model
        loss, losses, _ = run_pretrain_batch(batch, model, loss_fns, config['device'])
        
        # Update model parameters and step scheduler
        loss.backward()
        optim.step()
        scheduler.step()

        # Log the progress with wandb
        if batch_idx % config['log_every'] == 0 and wandblogger is not None:
            wandblogger.log({
                'batch': batch_idx,
                'combined_loss': loss.cpu().item(),
                'age_loss': losses[0].cpu().item(),
                'gender_loss': losses[1].cpu().item(),
                'seq_rec_loss': losses[2].cpu().item(),
                'learning_rate': float(optim.param_groups[0]['lr'])
            })
        
        # Save the model
        if batch_idx % config['save_every'] == 0 and batch_idx > 0 and wandblogger is not None:
            save_model(model, optim, scheduler, batch_idx, wandblogger.id)


def run_pretrain_batch(batch, model, losses, device):
    # Extract batch info
    signal, gender_y, age_y, mask, masked_seq = batch

    # Get the energy of signal for sequence reconstruction
    energy = torch.sum(signal ** 2, dim=-1) / signal.size(-1)
    energy = energy.unsqueeze(-1)

    # Send tensors to device
    signal = signal.to(device)
    masked_seq = masked_seq.to(device)
    mask = mask.to(device)
    gender_y = gender_y.to(device)
    age_y = age_y.to(device)
    energy = energy.to(device)

    # Run model masked and unmasked to get all results
    gender_pred, age_pred, _ = model(signal, None, None)
    _, _, seq_rec_pred = model(masked_seq, None, mask)

    # Get all the losses from all results for each task
    loss_gender = losses['gender'](gender_pred.squeeze(), gender_y.float())
    loss_age = losses['age'](age_pred.squeeze(), age_y.float())
    loss_seq_rec = losses['seq_rec'](seq_rec_pred, signal, mask)

    # Combine all losses by simply averaging
    loss = loss_gender

    # Returns losses and predictions
    return loss, [loss_age, loss_gender, loss_seq_rec], [gender_pred, age_pred, seq_rec_pred]
    

def finetune_epoch(dataloader, config, device, classifier, classifier_optim, scheduler, wandb_run):
    classifier.train()

    total_loss = []
    all_preds = []
    all_targets = []

    if config['classes'] == 1:
        classification_criterion = nn.BCEWithLogitsLoss()
    else:
        classification_criterion = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(dataloader):

        # Make sure we stop after the right amount of validation
        if batch_idx > config['batches_per_epoch']:
            break
        
        classifier_optim.zero_grad()

        if batch_idx % config['log_every'] == 0:
            logging.debug(f"Training batch {batch_idx}")
            print(f"Training batch {batch_idx}")

        loss, predictions, _ = simple_run_finetune_batch(batch, classifier, classification_criterion, config, device, True)

        # Optimize parameters
        loss.backward()
        nn.utils.clip_grad_value_(classifier.parameters(), config['clip'])

        if batch_idx == 0:
            plot_gradients = plot_grad_flow(classifier.cpu().named_parameters())
        classifier = classifier.to(device)

        classifier_optim.step()
        scheduler.step()

        with torch.no_grad():
            all_preds.append(predictions.detach().cpu())
            if config['full_transformer']:
                all_targets.append(batch[1][:, -1].squeeze(-1).detach())
            else:
                all_targets.append(batch[1].detach())
            total_loss.append(loss.cpu().item())
    
    with torch.no_grad():
        if config['classes'] == 1:
            acc, f1, recall, precision, cm = compute_metrics(torch.stack(
                all_preds, dim=0).to(device), torch.stack(all_targets, dim=0).to(device))
        else:
            metrics = classification_report(
                torch.stack(all_targets, dim=0).to(device),
                torch.stack(all_preds, dim=0).to(device),
                target_names=SleepStageDataset.get_labels()[:-1])
            print(metrics)
    
    if wandb_run is not None:
        wandb_run.log({
            'train/loss': torch.tensor(total_loss).mean().cpu().item(),
            'train/accuracy': acc,
            'train/F1': f1,
            'train/recall': recall,
            'train/precision': precision,
            'learning_rate': float(classifier_optim.param_groups[0]['lr']),
            'gradient_plot': plot_gradients
        })

    return torch.tensor(total_loss).mean(), acc, f1, recall, precision, cm, plot_gradients


def finetune_test_epoch(dataloader, config, classifier, device, wandb_run):
    with torch.no_grad():
        
        classifier.eval()

        all_preds = []
        all_targets = []
        total_loss = []

        if config['classes'] == 1:
            classification_criterion = nn.BCEWithLogitsLoss()
        else:
            classification_criterion = nn.CrossEntropyLoss()
        history, seqs = None, None
        for batch_idx, batch in enumerate(dataloader):
            
            # Stop early if we want to set a maximum validation length (usually for testing)
            if batch_idx > config['max_val_batches'] and config['max_val_batches'] != -1:
                break

            # Logging ever x epochs
            if batch_idx % config['log_every'] == 0:
                logging.debug(f"Validation batch {batch_idx}")
                print(f"Validation batch {batch_idx}")

            if batch_idx == 0:
                history = torch.zeros((batch[0].size(0), config['seq_len']-1)).to(device)

            # Run through model
            loss, predictions, seqs = simple_run_finetune_batch(batch, classifier, classification_criterion, config, device, False, seqs=seqs, history=history)

            if predictions is not None:
                history = history[:, 1:]
                history = torch.cat([history, predictions.unsqueeze(-1)], dim=1)
                all_preds.append(predictions.detach().cpu())
                all_targets.append(batch[1].detach())
                total_loss.append(loss.cpu().item())

        if config['classes'] == 1:
            acc, f1, recall, precision, cm = compute_metrics(torch.stack(
                all_preds, dim=0).to(device), torch.stack(all_targets, dim=0).to(device))
        else:
            metrics = classification_report(
                torch.stack(all_targets, dim=0).to(device),
                torch.stack(all_preds, dim=0).to(device),
                target_names=SleepStageDataset.get_labels()[:-1])

    if wandb_run is not None:
        wandb_run.log({'val/accuracy': acc, 'val/F1': f1, 'val/recall': recall, 'val/precision': precision})

    return torch.tensor(total_loss).mean(), acc, f1, recall, precision, cm


def simple_run_finetune_batch(batch, classifier, loss, config, device, training, seqs=None, history=None):
    # Extract info from batch
    seq, labels = batch

    # Get the full sequence from history of sequence
    if training or not config['full_transformer']:
        # Not necessary if training as batch already contains full sequence
        seqs = seq
    else:
        # Append current sequence to end of sequence
        seqs = torch.cat([seqs, seq.to(device)], dim=1) if seqs is not None else seq
        # Remove first in sequence if necessary
        if seqs.size(1) > config['seq_len']:
            seqs = seqs[:, 1:, :]      

    # Send tensors to device
    seqs = seqs.to(device)
    labels = labels.to(device)

    # Call model with accumulated sequence and history
    inferred = True
    if config['full_transformer']:
        if training:
            # If training, need to split labels to get both history and final label over whole sequence
            history, labels = labels.squeeze(-1).split(labels.size(1)-1, dim=-1)
            labels = labels.squeeze(-1)
        if seqs.size(1) == config['seq_len']:
            # Perform inferrence if sequence is long enough
            logits = classifier(seqs, history).squeeze(-1)
        else:
            inferred = False
    else:
        logits = classifier(seqs, None).squeeze(-1)

    # If we have performed inference, get predictions and loss and return them
    if inferred:
        # Convert logits to double if necessary
        # if config['classes'] != 1:
        #     logits = logits.type(torch.FloatTensor)
        loss = loss(logits, labels)
        if config['classes'] == 1:
            predictions = (torch.sigmoid(logits) > config['threshold']).int()
        else:
            predictions = torch.argmax(logits, dim=-1)
        return loss, predictions, seqs

    # We return None otherwise
    return None, None, seqs


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad): # and ("bias" not in n):
            if p.grad is None:
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.figure(figsize=(25, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = 0.0, top=max(max_grads)) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.draw()
    return wandb.Image(plt)


def compute_metrics(predictions, targets):
    predictions = predictions.type(torch.int)
    targets = targets.type(torch.int)

    tp = (targets * predictions)
    tn = ((1 - targets) * (1 - predictions))
    fp = ((1 - targets) * predictions)
    fn = (targets * (1 - predictions))

    tp_sum = tp.sum().to(torch.float32).item()
    fp_sum = fp.sum().to(torch.float32).item()
    fn_sum = fn.sum().to(torch.float32).item()
    tn_sum = tn.sum().to(torch.float32).item()
    confusion_matrix = {'True Positive': tp_sum, 'False Positive': fp_sum,
                        'False Negative': fn_sum, 'True Negative': tn_sum}
    epsilon = 1e-7

    precision = tp_sum / (tp_sum + fp_sum + epsilon)
    recall = tp_sum / (tp_sum + fn_sum + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    accuracy = (sum(tp) + sum(tn)) / (sum(tp) + sum(tn) + sum(fn) + sum(fp))

    return accuracy.mean(), f1, recall, precision, confusion_matrix


def count_shapes(tensor_list):
    init_shapes = {}
    for tensor in tensor_list:
        if str(tensor.shape) not in init_shapes.keys():
            init_shapes[str(tensor.shape)] = 0
        else:
            init_shapes[str(tensor.shape)] += 1
    return init_shapes

class WarmupTransformerLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, target_lr, decay, last_epoch=-1, verbose=False):
        if warmup_steps < 1:
            warmup_steps = 1
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.slope = target_lr / warmup_steps
        self.decay = decay
        super(WarmupTransformerLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        x = self.last_epoch + 1
        if x <= self.warmup_steps:
            return [(x * self.slope) for group in self.optimizer.param_groups]
        else:
            return [(group['lr'] * self.decay) for group in self.optimizer.param_groups]


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

