# CMPUT655-Project

## Installation Instructions

### Local Installation

```bash
conda create --name rl_project python=3.9
conda activate rl_project
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install jupyter
pip install "popgym[baselines]"
pip install tensorflow_probability==0.20.0
pip install mazelib
conda install matplotlib
cp custom_models/gtrxl.py ~/miniconda3/envs/rl_project/lib/python3.9/site-packages
```

### Google Colab

Run the following in a code cell before the rest of the project:

```bash
!pip install "popgym[baselines]"
!git clone https://github.com/john-science/mazelib.git
!pip install -r mazelib/requirements.txt
!pip install mazelib/
```
