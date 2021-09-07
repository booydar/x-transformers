from remote.variables import *

import torch, os
import numpy as np
import pandas as pd
from x_transformers.x_transformers import XTransformer
from torch.utils.tensorboard import SummaryWriter

class data_loader:
    def __init__(self, task_name, path='data', batch_size=32, enc_seq_len=16, dec_seq_len=16, none_mask=True):
        self.X, self.y = np.load(f'{path}/{task_name}_X.npy'), np.load(f'{path}/{task_name}_y.npy')
        self.data_size = self.X.shape[0]
        self.data_ptr = 0

        if none_mask:
            self.src_mask, self.tgt_mask = None, None
        else:
            self.src_mask = torch.ones(batch_size, enc_seq_len).bool().cuda()
            self.tgt_mask = torch.ones(batch_size, dec_seq_len+1).bool().cuda()
        self.batch_size = batch_size

    def __next__(self):
        X = self.X[self.data_ptr: self.data_ptr+self.batch_size]
        y = self.y[self.data_ptr: self.data_ptr+self.batch_size]
        self.data_ptr = (self.data_ptr + self.batch_size) % self.data_size

        return torch.tensor(X).cuda(), torch.tensor(y).cuda(), self.src_mask, self.tgt_mask

writer = SummaryWriter(log_dir='logs')


WINDOW_SIZE = 10
HEAD_START = 5
def train_validate_model(model, train_generator, val_generator, optim, model_name, generate_every=1e3, dec_seq_len=16, num_batches=1e4, verbose=True, overfit_stop=True, print_file=None):
    if print_file is None:
        print_file = f"logs/_{model_name}_cout_log.txt"

    validation_scores = []
    for i in range(num_batches):

        model.train()
        
        src, tgt, src_mask, tgt_mask = next(train_generator)
        loss = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        loss.backward()

        loss_value = loss.item()        
        writer.add_scalars("/train/loss", {model_name: loss_value}, i)
        if loss_value < 1e-10:
            break

        optim.step()
        optim.zero_grad()

        if i != 0 and i % generate_every == 0:
            model.eval()

            src, tgt, src_mask, _ = next(val_generator)
            tgt = tgt[:, 1:]
            start_tokens = tgt[:1, :1]
            
            if src_mask is not None:
                sm = src_mask[:1]
            else: 
                sm = src_mask

            num_correct = 0
            total_batch_len = 0
            for s, t in zip(src, tgt):
                sample = model.generate(s[None], start_tokens, dec_seq_len, src_mask=sm)
                num_correct += torch.abs(((t == sample) & (t != 0)).float()).sum()
                total_batch_len += (t != 0).float().sum()

            accuracy = num_correct / total_batch_len
            writer.add_scalars("/val/accuracy", {model_name: accuracy}, i)

            if verbose:
                with open(print_file, 'a') as f:
                    f.write(f"input:  {s}")
                    f.write(f"predicted output:  {sample}")
                    f.write(f"correct output:  {t}")
                    f.write(f"accuracy: {accuracy}")
                    
            if i // generate_every < HEAD_START:
                continue
            validation_scores.append(accuracy)
                
            # stop if val_acc drops
            if overfit_stop and max(validation_scores) > max(validation_scores[-WINDOW_SIZE:]):
                break

            

    writer.flush()


def test_model(model, test_generator, model_name, param, task_name, tag, dec_seq_len=16, log_path='logs/_test_results.csv'):
    model.eval()

    src, tgt, src_mask, _ = next(test_generator)
    tgt = tgt[:, 1:]
    start_tokens = tgt[:1, :1]
    if src_mask is not None:
        sm = src_mask[:1]
    else: 
        sm = src_mask

    num_correct = 0
    total_batch_len = 0
    for s, t in zip(src, tgt):
        sample = model.generate(s[None], start_tokens, dec_seq_len, src_mask=sm)
        num_correct += torch.abs(((t == sample) & (t != 0)).float()).sum()
        total_batch_len += (t != 0).float().sum()

    accuracy = num_correct / total_batch_len

    param['tag'] = tag
    param['task_name'] = task_name
    param['model_name'] = model_name
    param['accuracy'] = accuracy.cpu().item()

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = df.append(param, ignore_index=True)
    else: 
        df = pd.DataFrame([param])
    df.to_csv(log_path, index=False)