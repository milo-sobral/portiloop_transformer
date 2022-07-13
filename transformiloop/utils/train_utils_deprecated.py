from lib2to3.pytree import convert
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import wandb
from torchmetrics import F1Score, Accuracy
from sklearn.metrics import confusion_matrix, classification_report

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
            accuracy = 0
            counter = 0
            accuracy_scorer = Accuracy().to(config['device']) 
            for i in range(config['num_training_sets']):
                for batch_id, (source, _, _ , target) in enumerate(train_loader):
                    config["optimizer"].zero_grad()

                    source = source.squeeze(1)
                    prediction = model(source.to(config['device']))
                    target = target.unsqueeze(-1)
                    # Compute loss
                    loss = config["loss_func"](prediction, target.to(config['device']).type(torch.float))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
                    config["optimizer"].step()

                    # Compute loss
                    avg_loss += loss.item()
                    counter += 1
                    # Compute Accuracy
                    accuracy += accuracy_scorer(prediction.to(config['device']), target.to(config['device']).type(torch.int32))
                print(f"Current loss: {avg_loss / counter}")
                print(f"Current Accuracy: {accuracy / counter}")
                    
            train_loss = avg_loss / counter
            train_acc = accuracy / counter                        

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
            eval_loss, eval_acc, eval_precision, eval_recall, eval_f1 = eval_model_classification(model, config, val_loader)
            print(f"Losses at end of epoch: Train: {train_loss}, Eval: {eval_loss}")
            lr = config["scheduler"].get_last_lr()[0]
            print(f"Using learning rate: {lr}")
            stats = {'train_loss': train_loss, 
                     'train_acc': train_acc, 
                     'eval_loss':eval_loss, 
                     'lr': config["scheduler"].get_last_lr()[0], 
                     "eval_acc": eval_acc, 
                     "eval_precision": eval_precision,
                     "eval_recall": eval_recall,
                     'eval_f1': eval_f1}
            wandb.log(stats)

        config["scheduler"].step()


def eval_model_classification(model, config, val_loader):
    model.eval()
    avg_loss = 0
    num_correct = 0
    num_samples = 0
    counter = 0
    scorer = F1Score(num_classes=1, average='macro').to(config['device'])
    accuracy_scorer = Accuracy().to(config['device'])

    predictions, targets = None, None
    for batch_id, (source, _, _, target) in enumerate(val_loader):
        if batch_id > config['max_val_num']:
            break
        with torch.no_grad():
            source = source.squeeze(1).to(config['device'])
            target = target.unsqueeze(-1).to(config['device'])
            prediction = model(source)
            # Compute loss
            loss = config["loss_func"](prediction, target.type(torch.float))
            avg_loss += loss.item()
            counter += 1

            prediction = torch.sigmoid(prediction) > config['threshold']
            if predictions is None and targets is None:
                predictions = prediction
                targets = target
            else:
                predictions = torch.cat((predictions, prediction), 0)
                targets = torch.cat((targets, target), 0)
            # print(predictions.shape)
    predictions = predictions.type(torch.int)
    targets = targets.type(torch.int)

    tp = (targets * predictions)
    tn = ((1 - targets) * (1 - predictions))
    fp = ((1 - targets) * predictions)
    fn = (targets * (1 - predictions))

    f1_score, precision, recall = get_metrics(tp, fp, fn)

    # class_report = classification_report(predictions.type(torch.bool).tolist(), targets.type(torch.bool).tolist(), output_dict=True)

    final_loss = avg_loss / counter
    accuracy = (sum(tp) + sum(tn)) / (sum(tp) + sum(tn) + sum(fn) + sum(fp))
    # final_acc = accuracy_scorer(predictions, targets.type(torch.int32))
    # f1_score = scorer(predictions, targets.type(torch.int32).to(device=config['device']))
    # accuracy = class_report['accuracy']
    # precision = class_report['macro avg']['precision']
    # recall = class_report['macro avg']['recall']
    # f1_score = class_report['macro avg']['f1-score']

    return final_loss, accuracy, precision, recall, f1_score

def get_metrics(tp, fp, fn):
    tp_sum = tp.sum().to(torch.float32).item()
    fp_sum = fp.sum().to(torch.float32).item()
    fn_sum = fn.sum().to(torch.float32).item()
    epsilon = 1e-7

    precision = tp_sum / (tp_sum + fp_sum + epsilon)
    recall = tp_sum / (tp_sum + fn_sum + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1, precision, recall

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