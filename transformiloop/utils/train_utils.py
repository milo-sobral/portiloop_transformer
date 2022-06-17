from lib2to3.pytree import convert
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import wandb
from torchmetrics import F1Score, Accuracy

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks, device):
        super(MultiTaskLoss, self).__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros((num_tasks))).to(device)

    def forward(self, predictions, targets):
        assert len(predictions) == self.num_tasks, "Length of predictions must be number of tasks"
        assert len(targets) == self.num_tasks, "Length of targets must be number of tasks"
        mse = nn.MSELoss()

        losses = [mse(prediction, target) for (prediction, target) in zip(predictions, targets)]
        # final_losses = [torch.exp(-log_var) * loss + log_var for (log_var, loss) in zip(self.log_vars, losses)]
        final_losses = sum(losses) / self.num_tasks
        return final_losses


def train_model(model, config, train_loader, val_loader):
    print(f'Starting training for {config["epochs"]} epochs, using {config["device"]}.')
    for e in range(config['epochs']):
        print(f'\nEpoch {e+1}')

        model.to(config['device'])

        # Training loop
        model.train()

        if config['mode'] == 'pretraining':

            avg_loss = 0
            counter = 0
            for batch_id, (source, target_pred, target_rec) in enumerate(train_loader):
                config["optimizer"].zero_grad()

                prediction = model(source)

                if config["task"] == "both":
                    target = [target_pred, target_rec]
                elif config["task"] == "pred":
                    target = [target_pred]
                    prediction = [prediction[0]]
                elif config["task"] == "rec":
                    target = [target_rec]
                    prediction = [prediction[1]]

                loss = config["loss_func"](prediction, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
                config["optimizer"].step()
                
                if batch_id % config['log_every'] == 0:
                    print(f"Current loss: {loss.cpu().item()}")
                avg_loss += loss.cpu().item()
                counter += 1

            train_loss = avg_loss / counter
        elif config['mode'] == 'classification':
            avg_loss = 0
            num_correct = 0
            num_samples = 0
            counter = 0
            accuracy_scorer = Accuracy(num_classes=config['num_classes'], device=config['device']) 
            for batch_id, (source, _, _ , target) in enumerate(train_loader):
                config["optimizer"].zero_grad()

                source = source.squeeze(1)
                prediction = model(source.to(config['device']))
                target = convert_targets_to_logits(target).to(config['device'])
                # Compute loss
                loss = config["loss_func"](prediction, target.type(torch.float))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
                config["optimizer"].step()

                # Compute loss
                avg_loss += loss.item()
                counter += 1
                # Compute Accuracy
                accuracy = accuracy_scorer(prediction, target)

                if batch_id % config['log_every'] == 0:
                    print(f"Current loss: {loss.cpu().item()}")
                    print(f"Current Accuracy: {accuracy}")
            train_loss = avg_loss / counter
            train_acc = accuracy

        # Evaluate model
        if config['mode'] == 'pretraining':
            eval_loss = eval_model_pretraining(model, config, val_loader)
            print(f"Losses at end of epoch: Train: {train_loss}, Eval: {eval_loss}")
            lr = config["scheduler"].get_last_lr()[0]
            print(f"Using learning rate: {lr}")

            test_data, y_data_pred, y_data_rec = next(iter(val_loader))
            result_pred, result_rec = model(test_data[0].unsqueeze(0).cuda())
            plt.plot(torch.linspace(0, config['seq_len'], steps=config['seq_len']), y_data_rec[0].cpu(), label="Expected Recreate task")
            plt.plot(torch.linspace(0, config['seq_len'], steps=config['seq_len']), result_rec[0].cpu().detach(), label="Generated Recreate task")
            plt.plot(torch.arange(config['seq_len'], config['seq_len'] + config['out_seq_len']), y_data_pred[0].cpu(), label="Expected Predict task")
            plt.plot(torch.arange(config['seq_len'], config['seq_len'] + config['out_seq_len']), result_pred[0].cpu().detach(), label="Generated Predict task")
            plt.legend()
            wandb.log({'train_loss': train_loss, 'eval_loss':eval_loss, 'lr': config["scheduler"].get_last_lr()[0], "chart_viz": plt})
        elif config['mode'] == 'classification':
            eval_loss, eval_acc, eval_f1 = eval_model_classification(model, config, val_loader)
            print(f"Losses at end of epoch: Train: {train_loss}, Eval: {eval_loss}")
            lr = config["scheduler"].get_last_lr()[0]
            print(f"Using learning rate: {lr}")
            stats = {'train_loss': train_loss, 'train_acc': train_acc, 'eval_loss':eval_loss, 'lr': config["scheduler"].get_last_lr()[0], "eval_acc": eval_acc, 'eval_f1': eval_f1}
            wandb.log(stats)
            print(stats)

        config["scheduler"].step()


def eval_model_classification(model, config, val_loader):
    model.eval()
    avg_loss = 0
    num_correct = 0
    num_samples = 0
    counter = 0
    scorer = F1Score(num_classes=2, threshold=config['threshold'])
    accuracy_scorer = Accuracy(num_classes=config['num_classes']) 

    sources, targets = None, None
    for batch_id, (source, _, _, target) in enumerate(val_loader):
        with torch.no_grad():
            source = source.squeeze(1)
            prediction = model(source)
            target = convert_targets_to_logits(target)
            # Compute loss
            loss = config["loss_func"](prediction, target)
            avg_loss += loss.item()
            counter += 1
            # Compute Accuracy
            
            # Add to f1 score to measure
            if sources is None and targets is None:
                sources = source
                targets = target
            else:
                torch.cat(sources, source)
                torch.cat(targets, target)

    final_loss = avg_loss / counter
    final_acc = accuracy_scorer(sources, targets)
    f1_score = scorer(sources, targets)
    return final_loss, final_acc, f1_score

def eval_model_pretraining(model, config, val_loader):
    model.eval()
    avg_loss = 0
    counter = 0

    for batch_id, (source, target_pred, target_rec) in enumerate(val_loader):
        with torch.no_grad():
            prediction = model(source)

            if config["task"] == "both":
                target = [target_pred, target_rec]
            elif config["task"] == "pred":
                target = [target_pred]
                prediction = [prediction[0]]
            elif config["task"] == "rec":
                target = [target_rec]
                prediction = [prediction[1]]
                
            loss = config["loss_func"](prediction, target)
            avg_loss += loss.item()
            counter += 1
    final_loss = avg_loss / counter

    return final_loss

def convert_targets_to_logits(targets):
    logits = torch.zeros(targets.size(0), 2, dtype=torch.int)
    for idx, i in enumerate(targets):
        logits[idx, int(i)] = 1
    return logits