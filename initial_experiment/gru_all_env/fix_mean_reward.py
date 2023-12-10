# python initial_experiment/graphs.py ./initial_experiment <model-name>

import json
import matplotlib.pyplot as plt
import os
import sys


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    # path = sys.argv[1]
    # model_name = sys.argv[2]

    mean_reward_per_episode = load_json('./bad_mean_reward_per_episode.json')

    new_mean_reward_per_episode = {}
    size = len(mean_reward_per_episode['LabyrinthEscapeEasy'])
    for env in mean_reward_per_episode.keys():
        new_mean_reward_per_episode[env] = mean_reward_per_episode[env][:size]
        # plt.plot(mean_reward_per_episode[env])
        # plt.title(f'{env} - {model_name}')
        # plt.xlabel('Iterations')
        # plt.ylabel('Average Reward per Episode')
        # # plt.show()
        # plt.savefig(os.path.join(path, f'{env} - {model_name}.png'))
        # plt.clf()

    with open('./old_mean_reward_per_episode.json', 'w') as f:
        json.dump(new_mean_reward_per_episode, f, indent=4)
