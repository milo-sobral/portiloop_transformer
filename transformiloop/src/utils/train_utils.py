import torch
import numpy as np
from transformiloop.src.models.TFC.losses import NTXentLoss_poly
import torch.nn as nn
import os
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging
import wandb

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


def pretrain_epoch(model, model_optimizer, train_loader, config, device):
    total_loss = []
    losses_t = []
    losses_f = []
    losses_c = []
    model.train()

    for batch_idx, (data, aug1, data_f, aug1_f) in enumerate(train_loader):

        # Set up data
        data, aug1 = data.float().to(device), aug1.float().to(device)
        data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)

        # optimizer
        model_optimizer.zero_grad()

        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

        """Compute Pre-train loss"""
        """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
        nt_xent_criterion = NTXentLoss_poly(device, config['batch_size'], config['temperature'],
                                            config['use_cosine_similarity'])

        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)

        l_TF = nt_xent_criterion(z_t, z_f)
        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(
            z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        lam = config['lam']
        loss = lam * (loss_t + loss_f) + (1 - lam) * loss_c

        losses_t.append(loss_t.item())
        losses_f.append(loss_f.item())
        losses_c.append(loss_c.item())
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        if batch_idx % 100 == 0:
            print('pre-training: overall loss:{}, l_t: {}, l_f:{}, l_c:{}'.format(loss,
                  loss_t, loss_f, loss_c))
    total_loss = torch.tensor(total_loss).mean()
    loss_t = torch.tensor(losses_t).mean()
    loss_f = torch.tensor(losses_f).mean()
    loss_c = torch.tensor(losses_c).mean()

    return total_loss, loss_t, loss_f, loss_c

def finetune_epoch(model, model_optim, dataloader, config, device, classifier, classifier_optim):
    model.train()
    classifier.train()

    total_loss = []
    all_preds = []
    all_targets = []

    classification_criterion = nn.BCEWithLogitsLoss()

    for batch_idx, batch in enumerate(dataloader):
        
        model_optim.zero_grad()
        classifier_optim.zero_grad()

        if batch_idx % config['log_every'] == 0:
            logging.debug(f"Training batch {batch_idx}")
            print(f"Training batch {batch_idx}")

        loss, _, predictions = simple_run_finetune_batch(
            batch, classifier, classification_criterion, config['threshold'], config['lam'], device)

        # Optimize parameters
        loss.backward()

        if batch_idx == 0:
            plot_gradients = plot_grad_flow(classifier.cpu().named_parameters())
        classifier = classifier.to(device)


        model_optim.step()
        classifier_optim.step()

        with torch.no_grad():
            all_preds.append(predictions.detach().cpu())
            all_targets.append(batch[2])
            total_loss.append(loss.cpu().item())
    
    with torch.no_grad():
        acc, f1, recall, precision, cm = compute_metrics(torch.stack(
            all_preds, dim=0).to(device), torch.stack(all_targets, dim=0).to(device))
    # print(f"Accuracy: {acc*100}\nF1-score: {f1*100}\nRecall: {recall*100}\nPrecision: {precision*100}")
    return torch.tensor(total_loss).mean(), acc, f1, recall, precision, cm, plot_gradients


def finetune_test_epoch(model, dataloader, config, classifier, device):
    with torch.no_grad():
        
        model.eval()
        classifier.eval()

        all_preds = []
        all_targets = []
        total_loss = []

        classification_criterion = nn.BCEWithLogitsLoss()
        # nt_xent_criterion = NTXentLoss_poly(device, config['val_batch_size'], config['temperature'],
        #                                     config['use_cosine_similarity'])

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % config['log_every'] == 0:
                logging.debug(f"Validation batch {batch_idx}")
                print(f"Validation batch {batch_idx}")

            # Run through model
            loss, _, predictions = simple_run_finetune_batch(
                batch, classifier, classification_criterion, config['threshold'], config['lam'], device)
            all_preds.append(predictions.detach().cpu())
            all_targets.append(batch[2])
            total_loss.append(loss.cpu().item())

        acc, f1, recall, precision, cm = compute_metrics(torch.stack(
            all_preds, dim=0).to(device), torch.stack(all_targets, dim=0).to(device))
        # print(f"Accuracy: {acc*100}\nF1-score: {f1*100}\nRecall: {recall*100}\nPrecision: {precision*100}\nConfusion Matrix: {cm}")

    return torch.tensor(total_loss).mean(), acc, f1, recall, precision, cm


def simple_run_finetune_batch(batch, classifier, loss, threshold, lam, device):
    seqs, _, labels, _, _ = batch
    seqs = seqs.to(device)
    labels = labels.to(device)
    logits = classifier(seqs).squeeze(-1) 
    loss = loss(logits, labels)
    predictions = (torch.sigmoid(logits) > threshold).int()
    return loss, None, predictions


def run_finetune_batch(batch, model, classifier, model_loss, class_loss, threshold, lam, device):
    # loss_c = None
    # loss_t = None
    # loss_f = None

    encoded = []
    seqs, freqs, labels, seq_augs, freq_augs = batch
    seqs, freqs = seqs.float().to(device), freqs.float().to(device)
    seq_augs, freq_augs = seq_augs.float().to(device), freq_augs.float().to(device)

    # Run through encoder model
    for i in range(seqs.size(1)):
        seq = seqs[:, i, :].unsqueeze(1)
        freq = freqs[:, i, :].unsqueeze(1)
        # seq_aug = seq_augs[:, :, i, :]
        # freq_aug = freq_augs[:, :, i, :]

        h_t, z_t, h_f, z_f = model(seq, freq)
        # h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(seq_aug, freq_aug)

        # l_TF = model_loss(z_t, z_f)
        # l_1, l_2, l_3 = model_loss(z_t, z_f_aug), model_loss(
        #     z_t_aug, z_f), model_loss(z_t_aug, z_f_aug)

        # if loss_c is None:
        #     loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
        #     loss_t = model_loss(h_t, h_t_aug)
        #     loss_f = model_loss(h_f, h_f_aug)
        # else:
        #     loss_c += (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
        #     loss_t += model_loss(h_t, h_t_aug).item()
        #     loss_f += model_loss(h_f, h_f_aug).item()

        fea_concat = torch.cat((z_t, z_f), dim=1)
        encoded.append(fea_concat)

    # Run through classifier
    encoded = torch.stack(encoded, dim=1)
    logits = classifier(encoded).squeeze(-1)  # Run trhough transformer model
    # predictor loss, actually, here is training loss
    loss = class_loss(logits, labels.to(device))

    # Final loss taking into account all portions
    # loss_c /= seqs.size(1)
    # loss_t /= seqs.size(1)
    # loss_f /= seqs.size(1)
    # loss = loss_p + (1-lam) * loss_c + lam * (loss_t + loss_f)

    predictions = (torch.sigmoid(logits) > threshold).int()

    return loss, encoded, predictions

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
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


def nvidia_info():
    nvidia_smi.nvmlInit()

    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

    nvidia_smi.nvmlShutdown()