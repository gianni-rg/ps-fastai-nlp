# Setup Fast.AI 0.7 on Windows

The following steps, let you create an alternative Python environment (w.r.t. the one created in Module 3) where you can use Fast.AI v0.7 paired with PyTorch 0.4.1, in order to be able to follow the 2017-2018 v2 courses and examples by Jeremy Howard.

Once you have the pre-requisites ready, open a Command Prompt or a Powershell prompt and setup a new virtual environment for our Fast.AI 0.7 experiments.

1. If not already done, create a folder for our course projects

    `> cd C:\Projects`  
    `> mkdir ps-fastai-course`

2. Clone Fast.AI to its own folder

    `> cd C:\Projects\ps-fastai-course`  
    `> git clone https://github.com/fastai/fastai`

    This will create a `fastai` folder which contains the code for v1.0, but in the `old` subfolder you can find the 0.7 version, which we'll be using.

3. Create a Python virtual environment, `venv_psfastai07`

    `> python -m venv venv_psfastai07`

4. Activate it

    `> .\venv_psfastai07\scripts\activate`

5. Update pip, if not already done:

    `> python -m pip install --upgrade pip`

6. Install all required packages. Unfortunately, unlike with v1.0, we have to manually setup all the required dependencies.

    - Pytorch:

      Install PyTorch using one of the commands you can find on https://pytorch.org/:

        For GPU (CUDA 9.2):

        `> pip install https://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl`

        For CPU only:

        `> pip install https://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl`

        `> pip install torchvision`  
        `> pip install torchtext`

    - Jupyter:

        `> pip install jupyter`

    - Matplotlib:

        `> pip install matplotlib`

    - Scipy:

        `> pip install scipy`

    - Pandas:

        `> pip install pandas`

    - Seaborn:

        `> pip install seaborn`

    - Bcolz:

        `> pip install bcolz`

    - Scikit-learn:

        `> pip install scikit-learn`

    - Open-CV:

        `> pip install opencv-python`

    - Graphviz:

        `> pip install graphviz`

    - Sklearn Pandas:

        `> pip install sklearn_pandas`

    - Isoweek:

        `> pip install isoweek`

    - Pandas summary:

        `> pip install pandas_summary`

    - Spacy:

        `> pip install regex==2018.01.10`  
        `> pip install spacy`

    - Install ipython widgets into jupyter:

        `> jupyter nbextension enable --py widgetsnbextension --sys-prefix`

7. Create a subfolder for your Jupyter Notebooks with Fast.AI 0.7:

    `> mkdir nbs07`

8. Using and **Administrator** Command Prompt, create a symlink to Fast.AI folder:

    `> cd C:\Projects\ps-fastai-course\nbs07`  
    `> mklink /d fastai C:\Projects\fastai\old\fastai`

9. Create a symlink to Fast.AI `old\fastai` folder in each of the Fast.AI courses subfolders:

    `> cd C:\Projects\fastai\courses\dl1`  
    `> del fastai`  
    `> mklink /d fastai C:\Projects\fastai\old\fastai`

    `> cd C:\Projects\fastai\courses\dl2`  
    `> del fastai`  
    `> mklink /d fastai C:\Projects\fastai\old\fastai`

    `> cd C:\Projects\fastai\courses\ml1`  
    `> del fastai`  
    `> mklink /d fastai C:\Projects\fastai\old\fastai`

10. Download spaCy model:

    `> venv_psfastai07\scripts\activate`  
    `> python -m spacy download en`  

11. Close the Administrator Command Prompt, and go back in the Powershell prompt. Test the environment using the provided scripts (you can copy it from Module 3 materials):

    `> cd C:\Projects\ps-fastai-course\nbs07`  
    `> python env_test07.py`

    You should have a similar output:

            PyTorch version: 0.4.1
            Fast.AI version: 0.7
        PyTorch GPU-support: True

11. Then, let's launch a Jupyter notebook, and try to open one of the Fast.AI v2 courses notebook.

    `> cd C:\Projects\fastai\courses\dl1`  
    `> jupyter notebook`

    Open `Lesson1` or `Lesson4` and start executing the first cells, following all the steps, downloading the sample datasets... If everything is setup correctly, it will load and run without errors.

For additional information, or troubleshooting, you can refer to the Fast.AI forum threads:

- https://forums.fast.ai/t/howto-installation-on-windows/10439

- https://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652
