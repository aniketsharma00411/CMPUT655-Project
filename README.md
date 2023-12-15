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
### In the repository directory, there are three python notebook files :

1. Random_Agent_of_RL_env: Runs and saves the performance of a Random action agent on 100 seeds for all the required different environments.
   
2. pop_gym_env_exploration: Looks at some parts of the POPGym environments and prints out some detailed information for our exploration of the parts of the environments.

3. rl_project_experiment_structure: A basic code that trains on several environments and switches between them. (This code is not running the final experiments for that refer to **final_experiment folder**.

These files are supposed to be run in Google Collab and they perform various tasks not related to the training of the model on different tasks directly. 
These files 
