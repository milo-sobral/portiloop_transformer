import torch
import numpy as np
from transformiloop.models.TFC.losses import NTXentLoss_poly
import torch.nn as nn


def pretrain_epoch(model, model_optimizer, train_loader, config, device):
    total_loss = []
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
        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
        loss_c = (1+ l_TF -l_1) + (1+ l_TF -l_2) + (1+ l_TF -l_3)

        lam = 0.2
        loss = lam *(loss_t + loss_f) + (1- lam)*loss_c

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        if batch_idx % 100 == 0:
            print('pre-training: overall loss:{}, l_t: {}, l_f:{}, l_c:{}'.format(loss, loss_t, loss_f, loss_c)) 
    total_loss = torch.tensor(total_loss).mean()
    return total_loss


def finetune_epoch(model, model_optim, dataloader, config, device, classifier, classifier_optim, limit):
    model.train()
    classifier.train()

    total_loss = []
    all_preds = []
    all_targets = []

    classification_criterion = nn.BCEWithLogitsLoss()
    nt_xent_criterion = NTXentLoss_poly(device, config['batch_size'], config['temperature'],
                                        config['use_cosine_similarity'])

    for batch_idx, (seqs, freqs, labels, seq_augs, freq_augs) in enumerate(dataloader):
        if batch_idx > limit:
            break

        # Set up data
        seqs, freqs = seqs.float().to(device), freqs.float().to(device)
        seq_augs, freq_augs = seq_augs.float().to(device), freq_augs.float().to(device)

        model_optim.zero_grad()
        classifier_optim.zero_grad()

        loss_c = None

        encoded = []
        # Run through encoder model
        for i in range(seqs.size(2)):
            seq = seqs[:, :, i, :]
            freq = freqs[:, :, i, :]
            seq_aug = seq_augs[:, :, i, :]
            freq_aug = freq_augs[:, :, i, :]

            h_t, z_t, h_f, z_f = model(seq, freq)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug=model(seq_aug, freq_aug)

            loss_t = nt_xent_criterion(h_t, h_t_aug)
            loss_f = nt_xent_criterion(h_f, h_f_aug)
            l_TF = nt_xent_criterion(z_t, z_f)
            l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
                
            if loss_c is None:                                                                                             
                loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
            else:
                loss_c += (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

            fea_concat = torch.cat((z_t, z_f), dim=1)
            encoded.append(fea_concat)
        
        # Run through classifier
        encoded = torch.stack(encoded, dim=2) # ????? Dimension probably wrong here, need to stack based on dimension of sequence
        logits = classifier(encoded).squeeze(-1) # how to define classifier? MLP? CNN?
        loss_p = classification_criterion(logits, labels.to(device)) # predictor loss, actually, here is training loss

        # Final loss taking into account all portions
        loss_c /= seqs.size(2)
        lam = 0.2
        loss =  loss_p + (1-lam) * loss_c + lam * (loss_t + loss_f)

        predictions = (torch.sigmoid(logits) > config['threshold']).int()
        all_preds.append(predictions)
        all_targets.append(labels)

        # Optimize parameters
        loss.backward()
        model_optim.step()
        classifier_optim.step()
        total_loss.append(loss.cpu().item())

    acc, f1, recall, precision = compute_metrics(torch.stack(all_preds, dim=0).to(device), torch.stack(all_targets, dim=0).to(device))
    print(f"Accuracy: {acc*100}\nF1-score: {f1*100}\nRecall: {recall*100}\nPrecision: {precision*100}")
    return torch.tensor(total_loss).mean(), acc, f1, recall, precision

def finetune_test_epoch(model, dataloader, config, classifier, limit, device):
    model.eval()
    classifier.eval()

    all_preds = []
    all_targets = []

    for batch_idx, (seq, freq, labels, seq_aug, freq_aug) in enumerate(dataloader):
        if batch_idx > limit:
            break
        # Run throuhg model
        with torch.no_grad():
            seq, freq = seq.float().to(device), freq.float().to(device)
            seq_aug, freq_aug = seq_aug.float().to(device), freq_aug.float().to(device)

            _ , z_t, _, z_f = model(seq, freq)
            _, _, _, _ = model(seq_aug, freq_aug)

            # Run through classifier
            fea_concat = torch.cat((z_t, z_f), dim=1)
            logits = classifier(fea_concat).squeeze(-1) # how to define classifier? MLP? CNN?

            predictions = (torch.sigmoid(logits) > config['threshold']).int()       
            all_preds.append(predictions)
            all_targets.append(labels)

    acc, f1, recall, precision = compute_metrics(torch.stack(all_preds, dim=0).to(device), torch.stack(all_targets, dim=0).to(device))
    print(f"Accuracy: {acc*100}\nF1-score: {f1*100}\nRecall: {recall*100}\nPrecision: {precision*100}")

    return acc, f1, recall, precision


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
    epsilon = 1e-7

    precision = tp_sum / (tp_sum + fp_sum + epsilon)
    recall = tp_sum / (tp_sum + fn_sum + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    accuracy = (sum(tp) + sum(tn)) / (sum(tp) + sum(tn) + sum(fn) + sum(fp))

    return accuracy.mean(), f1, recall, precision

