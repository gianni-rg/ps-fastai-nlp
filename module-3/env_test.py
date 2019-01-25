#!/usr/bin/env python
# coding: utf-8

import fastai
import torch

if __name__ == '__main__':
    print(f'    PyTorch version: {torch.__version__}')
    print(f'    Fast.AI version: {fastai.__version__}')
    print(f'PyTorch GPU-support: {torch.cuda.is_available()}')
    