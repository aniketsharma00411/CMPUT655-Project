
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import PPOConfig
from popgym.envs import labyrinth_escape, labyrinth_explore
import torch
from popgym.baselines.ray_models.ray_gru import GRU
from popgym.baselines.ray_models.ray_lstm import LSTM
from popgym.baselines.ray_models.ray_linear_attention import DeepLinearAttention
import os
import pickle
import json
import sys

import pprint
import matplotlib.pyplot as plt
import time
# from ray.rllib.models.torch.attention_net import AttentionWrapper
"""# Configuration"""

num_of_cycles = 1
total_timesteps_per_cycle = 15e6

ray.init()

"""# Defining Environments"""

envs = ["LabyrinthEscapeEasy"]
        # , "LabyrinthEscapeMedium", "LabyrinthEscapeHard", "LabyrinthExploreEasy", "LabyrinthExploreMedium", "LabyrinthExploreHard"]

ray.tune.registry.register_env("LabyrinthEscapeEasy", lambda env_config: labyrinth_escape.LabyrinthEscapeEasy())
# ray.tune.registry.register_env("LabyrinthEscapeMedium", lambda env_config: labyrinth_escape.LabyrinthEscapeMedium())
# ray.tune.registry.register_env("LabyrinthEscapeHard", lambda env_config: labyrinth_escape.LabyrinthEscapeHard())
# ray.tune.registry.register_env("LabyrinthExploreEasy", lambda env_config: labyrinth_explore.LabyrinthExploreEasy())
# ray.tune.registry.register_env("LabyrinthExploreMedium", lambda env_config: labyrinth_explore.LabyrinthExploreMedium())
# ray.tune.registry.register_env("LabyrinthExploreHard", lambda env_config: labyrinth_explore.LabyrinthExploreHard())

"""# Defining Model"""

model = DeepLinearAttention



"""# Running Experiments"""

mean_reward_per_episode = {}
timesteps_done = {}
for env in envs:
    mean_reward_per_episode[env] = []
    timesteps_done[env] = []

previous_checkpoint_path = None

os.cpu_count()

total_timesteps = 0
prev_env_timesteps = 0
for cycle_count in range(1):
    for env in envs:
        print(f"Starting Cycle {cycle_count} Environment {env}:")
        # config = {

        num_splits = 1
        split_id = 1
        gpu_per_worker = 0.25
        # max_steps = 15e6
        # storage_path = os.environ.get("POPGYM_STORAGE", "/tmp/ray_results")
        # storage_path = "~/CMPUT655-Project/initial_experiment/ppo_gru/ray_results"
        # num_samples = 1

        # Used for testing
        # Maximum episode length and backprop thru time truncation length
        bptt_cutoff = 1024
        num_workers = 4
        num_minibatch = 8
        num_envs_per_worker = 16

        # Hidden size of linear layers
        h = 256
        # Hidden size of memory
        h_memory = 256
        train_batch_size = bptt_cutoff * max(num_workers, 1) * num_envs_per_worker
        config = {
        # Environments or env names
        "env": env,
        # Should always be torch
        "framework": "torch",
        # Number of rollout workers
        # "num_workers": num_workers,
        # # Number of envs per rollout worker
        # "num_envs_per_worker": num_envs_per_worker,
        # Num gpus used for the train worker
        "num_gpus": 0,
        # Loss coeff for the ppo value function
        "vf_loss_coeff": 1.0,
        # Num transitions in each training epoch
        "train_batch_size": train_batch_size,
        # Chunk size of transitions sent from rollout workers to trainer
        "rollout_fragment_length": bptt_cutoff,
        # Size of minibatches within epoch
        "sgd_minibatch_size": num_minibatch * bptt_cutoff,
        # decay gamma
        "gamma": 0.99,
        # Required due to RLlib PPO bugs
        "horizon": bptt_cutoff,
        # RLlib bug with truncate_episodes:
        # each batch the temporal dim shrinks by one
        # for now, just use complete_episodes
        "batch_mode": "complete_episodes",
        # "min_sample_timesteps_per_reporting": train_batch_size,
        "min_sample_timesteps_per_iteration": train_batch_size,
        # Describe your RL model here
        "model": {
            # Truncate sequences into no more than this many timesteps
            "max_seq_len": bptt_cutoff,
            # Custom model class
            "custom_model": model,
            # Config passed to custom model constructor
            # see base_model.py to see how these are used

            "custom_model_config": {
                "num_layers": 4,
                "preprocessor_input_size": h,
                "preprocessor": torch.nn.Sequential(
                    torch.nn.Linear(h, h),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                "preprocessor_output_size": h,
                "hidden_size": h_memory,
                "postprocessor": torch.nn.Identity(),
                "actor": torch.nn.Sequential(
                    torch.nn.Linear(h_memory, h),
                    torch.nn.LeakyReLU(inplace=True),
                    torch.nn.Linear(h, h),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                "critic": torch.nn.Sequential(
                    torch.nn.Linear(h_memory, h),
                    torch.nn.LeakyReLU(inplace=True),
                    torch.nn.Linear(h, h),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                "postprocessor_output_size": h,
            },
        },
    }

        trainer = PPOTrainer(env=env, config=config)
        # weights = trainer.get_weights()
        # with open('ray_torch_model.pkl', 'wb') as f:
        #     pickle.dump(weights, f)
        if previous_checkpoint_path is not None:
            trainer.restore(previous_checkpoint_path+"/"+sorted(os.listdir(previous_checkpoint_path))[-1])

        previous_checkpoint_path = f"{sys.argv[1]}/saved_checkpoints/agent_cycle_{cycle_count}_env_{env}"
        curr_env_timesteps = 0
        #variables to check whether algorithm has reached weight save points
        start_checkpoint = False
        midway_checkpoint = False

        start_time = time.time()
        while curr_env_timesteps < total_timesteps_per_cycle:
            result = trainer.train()

            mean_reward_per_episode[env].append(result["episode_reward_mean"])
            # pprint.pprint(result)
            total_timesteps = result["timesteps_total"]
            curr_env_timesteps = total_timesteps - prev_env_timesteps
            timesteps_done[env].append(curr_env_timesteps)

            # save weights when training checkpoints are reached
            # 1 represents start checkpoint and 2 is midway checkpoint
            if not start_checkpoint and curr_env_timesteps > 0:
                print("Saving weights: Timesteps ", curr_env_timesteps)
                with open(f"{sys.argv[1]}/saved_weights/weights_cycle_{cycle_count}_env_{env}_stage_1.pkl", "wb") as f:
                    weights = trainer.get_weights()
                    pickle.dump(weights, f)
                start_checkpoint = True

            if not midway_checkpoint and curr_env_timesteps >= total_timesteps_per_cycle / 2:
                print("Saving weights: Timesteps ", curr_env_timesteps)
                with open(f"{sys.argv[1]}/saved_weights/weights_cycle_{cycle_count}_env_{env}_stage_2.pkl", "wb") as f:
                    weights = trainer.get_weights()
                    pickle.dump(weights, f)
                midway_checkpoint = True

        trainer.save(previous_checkpoint_path)

        prev_env_timesteps = total_timesteps

        print("\n\n")
# print('Done')
ray.shutdown()