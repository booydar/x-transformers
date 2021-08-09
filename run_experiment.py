from .remote.variables import *

import torch, os
import numpy as np
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

def train_validate_model(model, train_generator, val_generator, optimizer, model_name, dec_seq_len=16, num_batches=1e4, verbose=True):

    for i in range(num_batches):

        model.train()
        
        src, tgt, src_mask, tgt_mask = next(train_generator)
        loss = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        loss.backward()

        loss_value = loss.item()
        if verbose:
            print(f'{i}: {loss_value}')
        
        writer.add_scalars("/train/loss", {model_name: loss_value}, i)
        if loss_value < 1e-10:
            break

        optim.step()
        optim.zero_grad()

        if i != 0 and i % GENERATE_EVERY == 0:
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
                print(f"input:  ", s)
                print(f"predicted output:  ", sample)
                print(f"correct output:  ", t)
                print(f"accuracy: {accuracy}")

    writer.flush()


def test_model(model, test_generator, model_name, param, dec_seq_len=16, log_path='logs/test_results.csv'):
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

    if not os.path.exists(log_path):
        with open(log_path, 'a') as f:
            f.write('task_name,')
            f.write('model_name,')
            for p in param:
                f.write(f'{p},')
            f.write('accuracy\n')

    with open(log_path, 'a') as f:
        f.write(f'{TASK_NAME},')
        f.write(f'{model_name},')
        for p in param:
            f.write(f'{param[p]},')
        f.write(f'{accuracy}\n')

    return accuracy

if __name__ == "__main__":
    gen_train = data_loader(task_name=f'{TASK_NAME}_train', batch_size=BATCH_SIZE, enc_seq_len=ENC_SEQ_LEN, dec_seq_len=DEC_SEQ_LEN)
    gen_val = data_loader(task_name=f'{TASK_NAME}_val', batch_size=VAL_SIZE, enc_seq_len=ENC_SEQ_LEN, dec_seq_len=DEC_SEQ_LEN)
    gen_test = data_loader(task_name=f'{TASK_NAME}_test', batch_size=TEST_SIZE, enc_seq_len=ENC_SEQ_LEN, dec_seq_len=DEC_SEQ_LEN)

    for param in list(model_parameters):
        print(param)
        for init_num in range(NUM_INITS):
            print(init_num)
            model = XTransformer(**param).cuda()

            model_name = f"{TASK_NAME}_dim{param['dim']}d{param['enc_depth']}h{param['enc_heads']}M{param['enc_num_memory_tokens']}l{ENC_SEQ_LEN}_v{init_num}"

            optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            train_validate_model(model, 
                                train_generator=gen_train, 
                                val_generator=gen_val, 
                                optimizer=optim, 
                                model_name=model_name, 
                                dec_seq_len=DEC_SEQ_LEN,
                                num_batches=NUM_BATCHES)
            test_model(model, gen_test, model_name, param, DEC_SEQ_LEN)
