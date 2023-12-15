# python initial_experiment/graphs_v1.py ./initial_experiment <model-name>

import json
import matplotlib.pyplot as plt
import os
import sys


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    # path = sys.argv[1]
    # model_name = sys.argv[2]

    model_names = {
        "gru_8_10_12": "GRU",
        "lstm3_8_10_12": "LSTM_3",
        "wfart_8_10_12": "WFART",
        "wgtrxl3_8_10_12": "WGTrXL_3"
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

    envs = ["LabyrinthEscapeVeryEasy", "LabyrinthEscapeEasy", "LabyrinthEscapeAlmostEasy",
            "LabyrinthExploreVeryEasy", "LabyrinthExploreEasy", "LabyrinthExploreAlmostEasy"]

    for env in envs:
        for model_name in ["gru_8_10_12", "lstm3_8_10_12", "wfart_8_10_12", "wgtrxl3_8_10_12"]:

            mean_reward_per_episode = load_json(os.path.join(
                'initial_experiment', model_name, "mean_reward_per_episode.json"))

            # for env in mean_reward_per_episode.keys():
            #     if len(mean_reward_per_episode[env]) == 0:
            #         continue

            timestamps, avg_reward = zip(*mean_reward_per_episode[env])

            plt.plot(timestamps, avg_reward,
                     label=f"${model_names[model_name]}$")

        plt.title(f"{env_names[env]}")
        plt.ylim((-1, 1))
        plt.xlim(xmin=0)
        plt.xlabel("Timesteps")
        plt.ylabel("Average Reward per Episode")
        plt.legend()
        plt.savefig(f"custom_graphs/graphs/{env_names[env]}.png")
        plt.clf()
