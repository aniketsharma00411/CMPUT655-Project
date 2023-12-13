
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

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
    os.makedirs(f"plots_for_weight_change", exist_ok=True)
    # plotting the total weight change l2 norm for the environment over all cycles and stages of training
    lst_of_wight_types = ["total ", "core", "actor", "critic", "preprocessor"]

    for index, x in enumerate([total_l2_norms, core_l2_norms, actor_l2_norms, critic_l2_norms, preprocessor_l2_norms]):

        # normalize the l2_norms with the first l2_norm in x to be the 1
        x = np.array(x)
        x = x / x[0]


        plt.plot(range(1, len(x) + 1), x, marker="o", markersize=3, linewidth=1)
        plt.xlabel("Visit Number")
        plt.ylabel("Weights")

        plt.xticks(range(1, len(x) + 1))
        #add a legend where the first argument is the label and the second is the color
    plt.title(f"Weight Change L2 Norm for {env}")
    plt.legend(lst_of_wight_types, loc="upper right")
    plt.savefig(f"plots_for_weight_change/weight_change_l2_norm_{env}.png")
    plt.close()


def init_weights(path):
    with open(path, "rb") as pth:
        weights = pickle.load(pth)
        # print("\t \t Types of weights found in the pickle file of numbers: ", len(weights["default_policy"].keys()))
        core, actor, critic, preprocessor, together = return_weights(weights)
        return {"core": core, "actor": actor, "critic": critic, "preprocessor": preprocessor, "together": together}


def main(cycle_count, weights_folder_pth, envs_list):
    for env in envs_list:
        env_l2_norm_dict_list = []


        for file in sorted(os.listdir(weights_folder_pth)):
            meta_data = file.split("_")
            env_name = meta_data[4]

            if env_name == env:
                for cycle in range(0, cycle_count):
                    prev_weights_dict = dict()
                    init_flag = True
                    for stage in [1, 2]:
                        pth = weights_folder_pth + "/" + f"weights_cycle_{cycle}_env_{env_name}_stage_{stage}.pkl"
                        with open(pth, "rb") as pth:
                            weights = pickle.load(pth)
                            print("for cycle:", cycle, "stage:", stage, "env:", env_name, ", weights loaded")
                            core, actor, critic, preprocessor, together = return_weights(weights)
                            curr_weight_dict = {"core": core, "actor": actor, "critic": critic,
                                                "preprocessor": preprocessor, "together": together}
                            if init_flag == False:
                                weight_change_l2_norms = calc_l2_norm(curr_weight_dict, prev_weights_dict)
                                for key in weight_change_l2_norms.keys():
                                    print(key, ":", weight_change_l2_norms[key])
                                print('curr_weights', curr_weight_dict)
                                print("prev weights", prev_weights_dict)
                                env_l2_norm_dict_list.append(weight_change_l2_norms)
                        prev_weights_dict = curr_weight_dict

                        if init_flag == True:
                            init_flag = False
                break
        plot_weight_change(env_l2_norm_dict_list, env)
if __name__ == '__main__':
    main(cycle_count=3,
         weights_folder_pth="/Users/srinjoybhuiya/PycharmProjects/CMPUT655-Project/final_experiment/gru/run2/saved_weights",
         envs_list=["LabyrinthEscapeAlmostEasy", "LabyrinthEscapeEasy", "LabyrinthEscapeVeryEasy"])


