import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import wandb

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks):
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
        avg_loss = 0
        counter = 0

        for batch_id, (source, target_pred, target_rec) in enumerate(train_loader):
            config["optimizer"].zero_grad()
            target = (target_pred, target_rec)
            prediction = model(source)
            loss = config["loss_func"](prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
            config["optimizer"].step()
            
            if batch_id % config['log_every'] == 0:
                print(f"Current loss: {loss.cpu().item()}")
            avg_loss += loss.cpu().item()
            counter += 1

        train_loss = avg_loss / counter

        # Evaluate model
        eval_loss = eval_model()

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

        config["scheduler"].step()


def eval_model(model, config, val_loader):
    model.eval()
    avg_loss = 0
    counter = 0

    for batch_id, (source, target_pred, target_rec) in enumerate(val_loader):
        with torch.no_grad():
            target = (target_pred, target_rec)
            prediction = model(source)
            loss = config["loss_func"](prediction, target)
            avg_loss += loss.item()
            counter += 1
    final_loss = avg_loss / counter

    return final_loss