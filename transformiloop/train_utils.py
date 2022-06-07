import time
import torch
from torch.autograd import Variable
import numpy as np


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=-99999):
        self.src = src
        # self.src_mask = (src != pad).unsqueeze(-2)
        self.src_mask = torch.ones(512, 64, 64).cuda()
        if trg is not None:
            self.trg = trg
            # self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    Ensures that decoder cannot see the future.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def rebatch(batch, batch_size, seq_len, take_y=False):
    (x, _, _, y) = batch
    x = torch.reshape(x, (batch_size, seq_len))
    src = Variable(x, requires_grad=False).cuda()
    if take_y:
        tgt = Variable(y, requires_grad=False).cuda()
    else:
        tgt = Variable(x, requires_grad=False).cuda()
    return Batch(src, trg=tgt)


def run_epoch(data_iter, model, loss_compute, batch_filter=None, batch_limit=None):
    "Standard Training and Logging Function"
    total_loss = 0
    
    for batch_count, batch in enumerate(data_iter):
        if batch_filter is not None:
            batch = batch_filter(batch)
        start_time = time.time()
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += float(loss)
        end_time = time.time()

        if batch_limit is not None:
            if batch_count >= batch_limit:
                break  

        if batch_count % 50 == 1:
            print("Batch #%d | Loss: %f | runtime: %fms" %
                    (batch_count, loss, end_time-start_time))
    return total_loss / batch_count


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1).unsqueeze(-1)) / norm

        # Check that we are training
        if self.opt is not None:
            loss.backward()
            self.opt.optimizer.zero_grad()
            self.opt.step()
        return loss.item() * norm