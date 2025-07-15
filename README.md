# Do transformers and recurrent neural networks lose plasticity in partially observable reinforcement learning tasks?

Reinforcement Learning project for [CMPUT 655: Reinforcement Learning I](https://apps.ualberta.ca/catalogue/course/cmput/655)

## Table of Contents
- [Abstract](#abstract)
- [Installation Instructions](#installation-instructions)
- [Repository Content](#repository-content)
- [Contributors](#contributors)

## Abstract

  
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
## Repository Content

In the repository directory, there are three Jupyter notebooks:

1. [Random_Agent_of_RL_env](Random_Agent_of_RL_env.ipynb): Runs and saves the performance of a Random action agent on 100 seeds for all the required different environments.
   
2. [pop_gym_env_exploration](pop_gym_env_exploration.ipynb): Looks at some parts of the POPGym environments and prints out some detailed information for our exploration of the parts of the environments.

3. [rl_project_experiment_structure](./rl_project_experiment_structure.ipynb): Python notebook defining the structure of the experiments performed. (For final experiments refer to [final_experiment](./final_experiment) directory).

Also, there are three directories:

1. [custom_models](./custom_models): Contains a modified GTRXL model code which we use in some initial experiments. 

2. [final_experiment](./final_experiment): Contains the final results of the GRU and FART(fast autoregressive transformers) based agent along with all our graphs and plots.

3. [initial_experiment](./initial_experiment): Contains results of all the several different models run during our initial testing

For each of the models in the final and initial experiments directories, we have an experiment.py script for model training and utility scripts for creating graphs.

## Contributors

- [Aniket Sharma](https://github.com/aniketsharma00411)
- [Harshil Kotamreddy](https://github.com/hk1510)
- [Manisimha Varma Manthena](https://github.com/Simha55)
- [Marcos Jose](https://github.com/MMenonJ)
- [Srinjoy Bhuiya](https://github.com/Srinjoycode)
