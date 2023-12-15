# python final_experiment/graphs.py ./final_experiment <model-name> <run>

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
    # run = sys.argv[3]

    model_names = {
        "gru_10_12_14": "GRU",
        "wfart_10_12_14": "WFART"
    }

    env_names = {
        "LabyrinthEscapeVeryEasy": "LabyrinthEscape_8",
        "LabyrinthEscapeEasy": "LabyrinthEscape_10",
        "LabyrinthEscapeAlmostEasy": "LabyrinthEscape_12",
        "LabyrinthEscapeMedium": "LabyrinthEscape_14",
        "LabyrinthEscapeHard": "LabyrinthEscape_18",
        "LabyrinthExploreVeryEasy": "LabyrinthExplore_8",
        "LabyrinthExploreEasy": "LabyrinthExplore_10",
        "LabyrinthExploreAlmostEasy": "LabyrinthExplore_12",
        "LabyrinthExploreMedium": "LabyrinthExplore_14",
        "LabyrinthExploreHard": "LabyrinthExplore_18"
    }

    envs = ["LabyrinthEscapeEasy", "LabyrinthEscapeAlmostEasy", "LabyrinthEscapeMedium",
            "LabyrinthExploreEasy", "LabyrinthExploreAlmostEasy", "LabyrinthExploreMedium"]

    for env in envs:
        for model_name in ["gru_10_12_14", "wfart_10_12_14"]:
            mean_reward_per_episode = load_json(os.path.join(
                'final_experiment', model_name, 'run1', 'mean_reward_per_episode.json'))
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

            plt.plot(modified_timestamps, avg_reward,
                     label=f"${model_names[model_name]}$")

        plt.title(f'{env_names[env]}')

        even = False
        for change_color_point in change_color_points:
            if even:
                plt.axvspan(
                    change_color_point[0], change_color_point[1], color='grey', alpha=0.3)
            even = not even

        plt.ylim((-1, 1))
        plt.xlim(xmin=0)
        plt.xlabel('Timesteps')
        plt.ylabel('Average Reward per Episode')
        plt.legend()
        plt.savefig(f'custom_graphs/graphs2/{env_names[env]}.png')
        plt.clf()
