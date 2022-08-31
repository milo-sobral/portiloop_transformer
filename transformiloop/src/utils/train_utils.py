import torch
import numpy as np
from transformiloop.src.models.TFC.losses import NTXentLoss_poly
import torch.nn as nn
import os
import json
import logging


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
        nt_xent_criterion = NTXentLoss_poly(device, config['batch_size'], config['encoder_config']['temperature'],
                                            config['encoder_config']['use_cosine_similarity'])

        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)

        l_TF = nt_xent_criterion(z_t, z_f)
        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(
            z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        lam = 0.3
        loss = lam * (loss_t + loss_f) + (1 - lam)*loss_c

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


def finetune_epoch(model, model_optim, dataloader, config, device, classifier, classifier_optim, limit):
    model.train()
    classifier.train()

    total_loss = []
    all_preds = []
    all_targets = []

    classification_criterion = nn.BCEWithLogitsLoss()
    nt_xent_criterion = NTXentLoss_poly(device, config['batch_size'], config['temperature'],
                                        config['use_cosine_similarity'])

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx > limit:
            break

        model_optim.zero_grad()
        classifier_optim.zero_grad()

        logging.debug(f"Training batch {batch_idx}")
        print(f"Training batch {batch_idx}")

        loss, _, predictions = run_finetune_batch(
            batch, model, classifier, nt_xent_criterion, classification_criterion, config['threshold'], config['lam'], device)

        all_preds.append(predictions)
        all_targets.append(batch[2])

        # Optimize parameters
        loss.backward()
        model_optim.step()
        classifier_optim.step()
        total_loss.append(loss.cpu().item())

    acc, f1, recall, precision, cm = compute_metrics(torch.stack(
        all_preds, dim=0).to(device), torch.stack(all_targets, dim=0).to(device))
    # print(f"Accuracy: {acc*100}\nF1-score: {f1*100}\nRecall: {recall*100}\nPrecision: {precision*100}")
    return torch.tensor(total_loss).mean(), acc, f1, recall, precision, cm


def finetune_test_epoch(model, dataloader, config, classifier, device, limit):
    model.eval()
    classifier.eval()

    all_preds = []
    all_targets = []
    total_loss = []

    classification_criterion = nn.BCEWithLogitsLoss()
    nt_xent_criterion = NTXentLoss_poly(device, config['val_batch_size'], config['temperature'],
                                        config['use_cosine_similarity'])

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx > limit:
            break

        logging.debug(f"Testing batch {batch_idx}")
        print(f"Testing batch {batch_idx}")

        # Run throuhg model
        with torch.no_grad():
            loss, _, predictions = run_finetune_batch(
                batch, model, classifier, nt_xent_criterion, classification_criterion, config['threshold'], config['lam'], device)
            all_preds.append(predictions)
            all_targets.append(batch[2])
            total_loss.append(loss.cpu().item())

    acc, f1, recall, precision, cm = compute_metrics(torch.stack(
        all_preds, dim=0).to(device), torch.stack(all_targets, dim=0).to(device))
    # print(f"Accuracy: {acc*100}\nF1-score: {f1*100}\nRecall: {recall*100}\nPrecision: {precision*100}\nConfusion Matrix: {cm}")

    return torch.tensor(total_loss).mean(), acc, f1, recall, precision, cm


def run_finetune_batch(batch, model, classifier, model_loss, class_loss, threshold, lam, device):
    loss_c = None
    loss_t = None
    loss_f = None

    encoded = []
    seqs, freqs, labels, seq_augs, freq_augs = batch
    seqs, freqs = seqs.float().to(device), freqs.float().to(device)
    seq_augs, freq_augs = seq_augs.float().to(device), freq_augs.float().to(device)

    # Run through encoder model
    for i in range(seqs.size(1)):
        # print(torch.cuda.memory_allocated(0))
        seq = seqs[:, i, :].unsqueeze(1)
        freq = freqs[:, i, :].unsqueeze(1)
        seq_aug = seq_augs[:, :, i, :]
        freq_aug = freq_augs[:, :, i, :]

        h_t, z_t, h_f, z_f = model(seq, freq)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(seq_aug, freq_aug)

        l_TF = model_loss(z_t, z_f)
        l_1, l_2, l_3 = model_loss(z_t, z_f_aug), model_loss(
            z_t_aug, z_f), model_loss(z_t_aug, z_f_aug)

        if loss_c is None:
            loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
            loss_t = model_loss(h_t, h_t_aug)
            loss_f = model_loss(h_f, h_f_aug)
        else:
            loss_c += (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
            loss_t += model_loss(h_t, h_t_aug).item()
            loss_f += model_loss(h_f, h_f_aug).item()

        fea_concat = torch.cat((z_t, z_f), dim=1)
        encoded.append(fea_concat)

    # Run through classifier
    encoded = torch.stack(encoded, dim=1)
    logits = classifier(encoded).squeeze(-1)  # Run trhough transformer model
    # predictor loss, actually, here is training loss
    loss_p = class_loss(logits, labels.to(device))

    # Final loss taking into account all portions
    loss_c /= seqs.size(1)
    loss_t /= seqs.size(1)
    loss_f /= seqs.size(1)
    loss = loss_p + (1-lam) * loss_c + lam * (loss_t + loss_f)

    predictions = (torch.sigmoid(logits) > threshold).int()

    return loss, encoded, predictions


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
