@echo off

REM Script to download e pre-process a Wikipedia dump.
REM Copyright (C) 2018-2019 Gianni Rosa Gallina. See LICENSE file.
REM Script ported to Windows batch file, based on Fast.AI Deep Learning Course v2 script 'prepare_wiki.sh'
REM and https://github.com/facebookresearch/fastText/blob/master/get-wikimedia.sh

Setlocal EnableDelayedExpansion

set ROOT=data
set DUMP_DIR=%ROOT%\wiki_dumps
set EXTR_DIR=%ROOT%\wiki_extr
set WIKI_DIR=%ROOT%\wiki
set EXTR=wikiextractor

IF NOT EXIST "%ROOT%" mkdir "%ROOT%"
IF NOT EXIST "%DUMP_DIR%" mkdir "%DUMP_DIR%"
IF NOT EXIST "%EXTR_DIR%" mkdir "%EXTR_DIR%"
IF NOT EXIST "%WIKI_DIR%" mkdir "%WIKI_DIR%"

echo Saving data in "%ROOT%"
set /P LANG="Choose a language (e.g. en, bh, fr, it, etc.): "

echo Chosen language: "%LANG%"

set DUMP_FILE=%LANG%wiki-latest-pages-articles.xml.bz2
set DUMP_PATH=%DUMP_DIR%\%DUMP_FILE%

IF NOT EXIST "%DUMP_PATH%" (
  set /P choice="Continue to download (WARNING: This might be big and can take a long time!) (y/n)? "
  IF /I "!choice!"=="y" (
    echo Starting download...
  ) ELSE IF /I "!choice!"=="n" (
    echo Exiting
    goto :eof
  ) ELSE (
    echo Invalid answer
    goto :eof
  )
  curl "https://dumps.wikimedia.org/%LANG%wiki/latest/%DUMP_FILE%" -o "%DUMP_PATH%"
) ELSE (
  echo "%DUMP_PATH%" already exists. Skipping download.
)

IF NOT EXIST "%EXTR%" (
  git clone https://github.com/attardi/wikiextractor.git
  cd "%EXTR%"
  python setup.py install
)

set EXTR_PATH=%EXTR_DIR%\%LANG%
IF NOT EXIST "%EXTR_PATH%" (
  set /P choice="Continue to extract Wikipedia (WARNING: This might take a long time!) (y/n)? "
  IF /I "!choice!"=="y" (
    echo Extracting "%DUMP_PATH%" to "%EXTR_PATH%"...
  ) ELSE IF /I "!choice!"=="n" (
    echo Exiting
    goto :eof
  ) ELSE (
    echo Invalid answer
    goto :eof
  )
  python wikiextractor/WikiExtractor.py -s --json -o "%EXTR_PATH%" "%DUMP_PATH%"
) ELSE (
  echo "%EXTR_PATH%" already exists. Skipping extraction.
)

set OUT_PATH=%WIKI_DIR%\%LANG%
set /P choice="Continue to merge Wikipedia articles (y/n)? "
IF /I "!choice!"=="y" (
    echo Merging articles from "%EXTR_PATH%" to "%OUT_PATH%"...
  ) ELSE IF /I "!choice!"=="n" (
    echo Exiting
    goto :eof
  ) ELSE (
    echo Invalid answer
    goto :eof
  )
python merge_wiki.py -i "%EXTR_PATH%" -o "%OUT_PATH%"
