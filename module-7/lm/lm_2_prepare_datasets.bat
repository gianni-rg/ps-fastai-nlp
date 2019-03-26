@echo off

REM Script to prepare datasets for Custom Language Model Training
REM Copyright (C) 2018-2019 Gianni Rosa Gallina. See LICENSE file.

Setlocal EnableDelayedExpansion

set ROOT=data
set WIKI_DIR=%ROOT%/wiki

REM Change the following settings accordingly, to train a model for
REM a desired language with the specified vocabulary size

set LANG=it
set MAX_VOCAB=60000
set MIN_FREQ=2

echo Working with data from "%WIKI_DIR%/%LANG%"

echo Tokenizing and Numericalizing text...
python preprocess_text.py --dir_path "%WIKI_DIR%/%LANG%" --lang %LANG% --max_vocab %MAX_VOCAB% --min_freq %MIN_FREQ%
