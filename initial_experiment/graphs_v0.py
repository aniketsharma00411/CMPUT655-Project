# python initial_experiment/graphs_v0.py ./initial_experiment <model-name>

import json
import matplotlib.pyplot as plt
import os
import sys


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    path = sys.argv[1]
    model_name = sys.argv[2]

    model_names = {
        "gru": "GRU",
        "gru_10_14_18": "GRU",
        "lstm1": "LSTM_1",
        "lstm2": "LSTM_2",
        "ngtrxl1": "NGTrXL_1",
        "wgtrxl1": "WGTrXL_1",
        "wgtrxl2": "WGTrXL_2"
    }

    timesteps = {
        "gru": (65536, int(15e6)),
        "gru_10_14_18": (65536, int(15e6)),
        "lstm1": (4000, int(1e6)+4000),
        "lstm2": (4000, int(1e6)+4000),
        "ngtrxl1": (4000, int(1e6)+4000),
        "wgtrxl1": (4000, int(1e6)+4000),
        "wgtrxl2": (4000, int(1e6)+4000)
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

    mean_reward_per_episode = load_json(os.path.join(
        path, model_name, "mean_reward_per_episode.json"))

    for env in mean_reward_per_episode.keys():
        plt.plot(range(timesteps[model_name][0], timesteps[model_name]
                 [1], timesteps[model_name][0]), mean_reward_per_episode[env])
        plt.title(f"{env_names[env]} - ${model_names[model_name]}$")
        plt.ylim((-1, 1))
        plt.xlim(xmin=0)
        plt.xlabel("Timesteps")
        plt.ylabel("Average Reward per Episode")
        plt.savefig(os.path.join(path, model_name, "graphs",
                    f"{env_names[env]} - {model_names[model_name]}.png"))
        plt.clf()
