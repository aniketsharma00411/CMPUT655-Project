# python final_experiment/graphs.py ./final_experiment <model-name> <run>

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
    run = sys.argv[3]

    mean_reward_per_episode = load_json(os.path.join(
        path, model_name, run, 'mean_reward_per_episode.json'))

    for env in mean_reward_per_episode.keys():
        if len(mean_reward_per_episode[env]) == 0:
            continue

        timestamps, avg_reward = zip(*mean_reward_per_episode[env])

        modified_timestamps = []
        change_color_points = []
        last_cycle_timestamp = 0
        last_timestamp = 0
        for timestamp in timestamps:
            if timestamp < last_timestamp:
                change_color_points.append(
                    (last_cycle_timestamp, last_timestamp+last_cycle_timestamp))
                last_cycle_timestamp = last_timestamp+last_cycle_timestamp

            modified_timestamps.append(timestamp+last_cycle_timestamp)
            last_timestamp = timestamp
        change_color_points.append(
            (last_cycle_timestamp, last_timestamp+last_cycle_timestamp))

        plt.plot(modified_timestamps, avg_reward)
        plt.title(f'{env} - {model_name}')

        even = False
        for change_color_point in change_color_points:
            if even:
                plt.axvspan(
                    change_color_point[0], change_color_point[1], color='grey', alpha=0.3)
            even = not even

        plt.ylim((-1, 1))
        plt.xlabel('Timesteps')
        plt.ylabel('Average Reward per Episode')
        plt.savefig(os.path.join(path, model_name, run,
                    f'graphs/{env} - {model_name}.png'))
        plt.clf()
