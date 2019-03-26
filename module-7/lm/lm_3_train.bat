@echo off

REM Script to train a Custom Language Model
REM Copyright (C) 2018-2019 Gianni Rosa Gallina. See LICENSE file.

Setlocal EnableDelayedExpansion

set ROOT=data
set WIKI_DIR=%ROOT%/wiki

REM Change the following settings accordingly, to train a model for
REM a desired language with the specificied settings. Those settings
REM must be tweaked to find the best combination for a specific
REM language, vocabulary size and training hardware.

set LANG=it
set CUDA_DEVICE=0
set MAX_LR=5e-3
set CYCLE_LEN=15
set BATCH_SIZE=32

echo Working with data from "%WIKI_DIR%/%LANG%"

python pretrain_lm_v1.py --dir_path "%WIKI_DIR%/%LANG%" --cuda_id %CUDA_DEVICE% --lr %MAX_LR% --cl %CYCLE_LEN% --bs %BATCH_SIZE%