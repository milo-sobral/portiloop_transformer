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

def save_model(save_path, model, config):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    not_saveable_keys = ['device', 'optimizer',
                         'scheduler', 'loss_func', 'classifier_optimizer']
    saveable_config = {key: val for (
        key, val) in config.items() if key not in not_saveable_keys}
    print(saveable_config)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(saveable_config, f)
    torch.save(model.state_dict(), os.path.join(save_path, "model"))


def seq_rec_loss(predictions, expected, mask):
    loss = nn.MSELoss()
    mask = torch.where(mask==0, mask, 1).unsqueeze(-1)
    mask = mask.expand(mask.size(0), mask.size(1), predictions.size(-1))
    return loss(predictions * mask, expected * mask)
    

def pretrain_epoch(dataloader, config, device, model, optim, scheduler):
    model.train()

    # TODO: Add the metrics for all tasks to keep track of training
    mse_loss = nn.MSELoss()
    losses = {
        'gender': nn.BCEWithLogitsLoss(),
        'age': mse_loss,
        'seq_rec': (lambda pred, exp : seq_rec_loss(pred, exp, mse_loss))
    }

    for batch_idx, batch in enumerate(dataloader):
        optim.zero_grad()
        if batch_idx % config['log_every'] == 0:
            logging.debug(f"Training batch {batch_idx}")

        loss, _, predictions = run_pretrain_batch(batch, model, losses)

        loss.backwards()
        optim.step()
        scheduler.step()


def run_pretrain_batch(batch, model, losses):
    # Extract batch info
    signal, gender_y, age_y, mask, masked_seq = batch

    # Run model masked and unmasked to get all results
    gender_pred, age_pred, _ = model(signal, None, None)
    _, _, seq_rec_pred = model, masked_seq, None, mask

    # Get all the losses from all results for each task
    loss_gender = losses['gender'](gender_pred, gender_y)
    loss_age = losses['age'](age_pred, age_y)
    loss_seq_rec = losses['seq_rec'](seq_rec_pred, signal, mask)

    # Combine all losses by simply averaging
    loss = (loss_age + loss_gender + loss_seq_rec) / 3.0

    # Returns losses and predictions
    return loss, [loss_age, loss_gender, loss_seq_rec], [gender_pred, age_pred, seq_rec_pred]
    


def finetune_epoch(dataloader, config, device, classifier, classifier_optim, scheduler):
    classifier.train()

    total_loss = []
    all_preds = []
    all_targets = []

    classification_criterion = nn.BCEWithLogitsLoss()

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
        acc, f1, recall, precision, cm = compute_metrics(torch.stack(
            all_preds, dim=0).to(device), torch.stack(all_targets, dim=0).to(device))
    # print(f"Accuracy: {acc*100}\nF1-score: {f1*100}\nRecall: {recall*100}\nPrecision: {precision*100}")
    return torch.tensor(total_loss).mean(), acc, f1, recall, precision, cm, plot_gradients


def finetune_test_epoch(dataloader, config, classifier, device):
    with torch.no_grad():
        
        classifier.eval()

        all_preds = []
        all_targets = []
        total_loss = []

        classification_criterion = nn.BCEWithLogitsLoss()
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

        acc, f1, recall, precision, cm = compute_metrics(torch.stack(
            all_preds, dim=0).to(device), torch.stack(all_targets, dim=0).to(device))

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
        loss = loss(logits, labels)
        predictions = (torch.sigmoid(logits) > config['threshold']).int()
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


# DEPRECATED
# def run_finetune_batch(batch, model, classifier, class_loss, threshold, lam, device):
#     # loss_c = None
#     # loss_t = None
#     # loss_f = None

#     encoded = []
#     seqs, freqs, labels, seq_augs, freq_augs = batch
#     seqs, freqs = seqs.float().to(device), freqs.float().to(device)
#     seq_augs, freq_augs = seq_augs.float().to(device), freq_augs.float().to(device)

#     # Run through encoder model
#     for i in range(seqs.size(1)):
#         seq = seqs[:, i, :].unsqueeze(1)
#         freq = freqs[:, i, :].unsqueeze(1)
#         # seq_aug = seq_augs[:, :, i, :]
#         # freq_aug = freq_augs[:, :, i, :]

#         h_t, z_t, h_f, z_f = model(seq, freq)
#         # h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(seq_aug, freq_aug)

#         # l_TF = model_loss(z_t, z_f)
#         # l_1, l_2, l_3 = model_loss(z_t, z_f_aug), model_loss(
#         #     z_t_aug, z_f), model_loss(z_t_aug, z_f_aug)

#         # if loss_c is None:
#         #     loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
#         #     loss_t = model_loss(h_t, h_t_aug)
#         #     loss_f = model_loss(h_f, h_f_aug)
#         # else:
#         #     loss_c += (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
#         #     loss_t += model_loss(h_t, h_t_aug).item()
#         #     loss_f += model_loss(h_f, h_f_aug).item()

#         fea_concat = torch.cat((z_t, z_f), dim=1)
#         encoded.append(fea_concat)

#     # Run through classifier
#     encoded = torch.stack(encoded, dim=1)
#     logits = classifier(encoded).squeeze(-1)  # Run trhough transformer model
#     # predictor loss, actually, here is training loss
#     loss = class_loss(logits, labels.to(device))

#     # Final loss taking into account all portions
#     # loss_c /= seqs.size(1)
#     # loss_t /= seqs.size(1)
#     # loss_f /= seqs.size(1)
#     # loss = loss_p + (1-lam) * loss_c + lam * (loss_t + loss_f)

#     predictions = (torch.sigmoid(logits) > threshold).int()

#     return loss, encoded, predictions

# def nvidia_info():
#     nvidia_smi.nvmlInit()

#     deviceCount = nvidia_smi.nvmlDeviceGetCount()
#     for i in range(deviceCount):
#         handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
#         info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
#         print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

#     nvidia_smi.nvmlShutdown()