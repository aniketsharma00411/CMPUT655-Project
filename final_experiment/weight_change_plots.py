# python final_experiment/weight_change_plots.py ./final_experiment <model-name> <run>

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def return_weights(weights):
    core = []
    actor = []
    critic = []
    preprocessor = []
    together = []
    for key in weights["default_policy"].keys():
        # print("\t", key, "||| of shape:", weights["default_policy"][key].shape)
        weights_type = key.split(".")[0]
        if weights_type == "core" or weights_type == "unmap":
            core.append(weights["default_policy"][key].flatten())
        if weights_type == "actor":
            actor.append(weights["default_policy"][key].flatten())
        if weights_type == "critic":
            critic.append(weights["default_policy"][key].flatten())
        if weights_type != "critic" and weights_type != "actor" and weights_type != "core" and weights_type != "unmap":
            preprocessor.append(weights["default_policy"][key].flatten())
        # for all weights totaled together
        together.append(weights["default_policy"][key].flatten())

    return np.concatenate(core), np.concatenate(actor), np.concatenate(critic), np.concatenate(
        preprocessor), np.concatenate(together)


def calc_l2_norm(w1_dict, w2_dict):
    print("calculating l2 norm")
    l2_norm_dict = dict()
    for key in w1_dict.keys():
        l2_norm_dict[key] = np.linalg.norm(w1_dict[key] - w2_dict[key])
    return l2_norm_dict


def plot_weight_change(env_l2_norm_dict_list, env):
    total_l2_norms = []
    core_l2_norms = []
    actor_l2_norms = []
    critic_l2_norms = []
    preprocessor_l2_norms = []

    model_names = {
        "gru_8_10_12": "GRU",
        "gru_10_12_14": "GRU",
        "wfart_8_10_12": "WFART",
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

    for l2_norm in env_l2_norm_dict_list:
        for key in l2_norm.keys():
            if key == "core":
                core_l2_norms.append(l2_norm[key])
            if key == "actor":
                actor_l2_norms.append(l2_norm[key])
            if key == "critic":
                critic_l2_norms.append(l2_norm[key])
            if key == "preprocessor":
                preprocessor_l2_norms.append(l2_norm[key])
            if key == "together":
                total_l2_norms.append(l2_norm[key])

    # plotting the total weight change l2 norm for the environment over all cycles and stages of training
    lst_of_wight_types = ["total ", "core", "actor", "critic", "preprocessor"]

    for _, x in enumerate([total_l2_norms, core_l2_norms, actor_l2_norms, critic_l2_norms, preprocessor_l2_norms]):
        # normalize the l2_norms with the first l2_norm in x to be the 1
        x = np.array(x)
        x = x / x[0]

        plt.plot(range(1, len(x) + 1), x, marker="o",
                 markersize=3, linewidth=1)
        plt.xlabel("Visit")
        plt.ylabel("Normalized $l_2$ norm")

        plt.xticks(range(1, len(x) + 1))

    plt.title(f"{env_names[env]} - ${model_names[model_name]}$")
    plt.ylim((0, 2.2))
    plt.xlim(xmin=1)
    plt.legend(lst_of_wight_types)
    plt.savefig(os.path.join(
        path, model_name, run, "weight_change_graphs", f"weight_{env_names[env]}_{model_names[model_name]}.png"))
    plt.close()


def init_weights(path):
    with open(path, "rb") as pth:
        weights = pickle.load(pth)
        # print("\t \t Types of weights found in the pickle file of numbers: ", len(weights["default_policy"].keys()))
        core, actor, critic, preprocessor, together = return_weights(weights)
        return {"core": core, "actor": actor, "critic": critic, "preprocessor": preprocessor, "together": together}


def main(num_of_cycles, weights_folder_pth, envs):
    for env in envs:
        env_l2_norm_dict_list = []

        for file in sorted(os.listdir(weights_folder_pth)):
            meta_data = file.split("_")
            env_name = meta_data[4]

            if env_name == env:
                for cycle in range(0, num_of_cycles):
                    prev_weights_dict = dict()
                    init_flag = True
                    for stage in [1, 2]:
                        pth = weights_folder_pth + "/" + \
                            f"weights_cycle_{cycle}_env_{env_name}_stage_{stage}.pkl"
                        with open(pth, "rb") as pth:
                            weights = pickle.load(pth)
                            print("for cycle:", cycle, "stage:", stage,
                                  "env:", env_name, ", weights loaded")
                            core, actor, critic, preprocessor, together = return_weights(
                                weights)
                            curr_weight_dict = {"core": core, "actor": actor, "critic": critic,
                                                "preprocessor": preprocessor, "together": together}
                            if init_flag == False:
                                weight_change_l2_norms = calc_l2_norm(
                                    curr_weight_dict, prev_weights_dict)
                                for key in weight_change_l2_norms.keys():
                                    print(
                                        key, ":", weight_change_l2_norms[key])
                                print('curr_weights', curr_weight_dict)
                                print("prev weights", prev_weights_dict)
                                env_l2_norm_dict_list.append(
                                    weight_change_l2_norms)
                        prev_weights_dict = curr_weight_dict

                        if init_flag == True:
                            init_flag = False
                break

        plot_weight_change(env_l2_norm_dict_list, env)


if __name__ == '__main__':
    path = sys.argv[1]
    model_name = sys.argv[2]
    run = sys.argv[3]

    envs = ["LabyrinthExploreEasy", "LabyrinthEscapeMedium", "LabyrinthExploreAlmostEasy",
            "LabyrinthEscapeEasy", "LabyrinthExploreMedium", "LabyrinthEscapeAlmostEasy"]
    # envs = ["LabyrinthExploreEasy", "LabyrinthEscapeVeryEasy", "LabyrinthExploreAlmostEasy",
    #         "LabyrinthEscapeEasy", "LabyrinthExploreVeryEasy", "LabyrinthEscapeAlmostEasy"]
    main(
        num_of_cycles=10,
        weights_folder_pth=os.path.join(
            path, model_name, run, "saved_weights"),
        envs=envs
    )
