# Copyright (C) 2018-2019 Gianni Rosa Gallina. See LICENSE file.
# Partially based on Fast.AI Deep Learning Course v2 scripts.

from fastai.text import *
from fastai.callbacks import CSVLogger, SaveModelCallback
import argparse

def train_lm(dir_path, cuda_id, cycle_len=30, bs=64, max_lr=3e-4):
    print(f'   dir_path: {dir_path}')
    print(f'    cuda_id: {cuda_id}')
    print(f'  cycle-len: {cycle_len}')
    print(f' batch-size: {bs}')
    print(f'     max_lr: {max_lr}')

    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1

    torch.cuda.set_device(cuda_id)
    torch.cuda.empty_cache()
    
    dir_path = Path(dir_path)
    assert dir_path.exists(), f'Error: {dir_path} does not exist.'
    
    tmp_path = dir_path/'tmp'
    tmp_path.mkdir(exist_ok=True)
    
    data = TextLMDataBunch.load(dir_path,bs=bs)
    
    print("Data loaded")
    
    learn = language_model_learner(data, bptt = 70, emb_sz = 400, nh = 1150, nl = 3,
                               drop_mult = 0.05, alpha = 2, beta = 1, clip = 0.12, wd = 1e-7)
    learn.opt_func = partial(optim.Adam, betas=(0.8, 0.99))
    learn.callback_fns += [partial(CSVLogger, filename=f"{learn.model_dir}/lm-history"),
                           partial(SaveModelCallback, every='epoch', name='lm')]
 
    lr = slice(max_lr/(2.6**4), max_lr)
 
    print("Learner configured")
        
    learn.fit_one_cycle(cyc_len = cycle_len,# Number of Epochs
                        max_lr = lr,        # Learning rate
                        div_factor = 20,    # Factor to discount from max
                        moms = (0.8, 0.7),  # Momentums
                        pct_start = 0.1,    # Where the peak is at
                        ) 
    
    learn.save_encoder('enc_lstm')
    learn.save('model_lstm')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', required=True,
                        help='the directory where the merged Wikipedia '
                             'documents are stored')
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='the CUDA device to use')
    parser.add_argument('--cl', type=int, default=30,
                        help='the number of epochs to run')
    parser.add_argument('--bs', type=int, default="64",
                        help='the number of text documents to load in a batch')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='the maximum learning rate to use')
    args = parser.parse_args()

    train_lm(args.dir_path, args.cuda_id, args.cl, args.bs, args.lr)
