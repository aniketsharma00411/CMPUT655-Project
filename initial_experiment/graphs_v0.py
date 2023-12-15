# python initial_experiment/graphs_v0.py ./initial_experiment <model-name>

import json
import matplotlib.pyplot as plt
import os
import sys


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    path = sys.argv[1]
    model_name = sys.argv[2]

    mean_reward_per_episode = load_json(os.path.join(
        path, model_name, 'mean_reward_per_episode.json'))

    for env in mean_reward_per_episode.keys():
        plt.plot(range(65536, int(15e6), 65536), mean_reward_per_episode[env])
        plt.title(f'{env} - {model_name}')
        plt.xlabel('Timesteps')
        plt.ylabel('Average Reward per Episode')
        plt.savefig(os.path.join(path, model_name, 'graphs',
                    f'{env} - {model_name}.png'))
        plt.clf()
