import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

weights_folder_pth="gru_sample_weights"


flag_init = True

total_l2_norms=[]
actor_l2_norms=[]
critic_l2_norms=[]
preprocessor_l2_norms=[]
core_l2_norms=[]

prev_weights_list = []
curr_weight=[]

preprocessor_curr_list=[]
preprocessor_prev_list=[]

core_curr_list=[]
core_prev_list=[]

actor_curr_list=[]
actor_prev_list=[]

critic_curr_list=[]
critic_prev_list=[]

for cycle_num in range(len(os.listdir(weights_folder_pth))):

    print("For Cycle :", cycle_num)

    with open(f"{weights_folder_pth}/weights_{cycle_num}_test.pkl", "rb") as f:
        weights = pickle.load(f)

        print("\t Types of weights found in the pickle file of numbers: ", len(weights["default_policy"].keys()))

        for key in weights["default_policy"].keys():
            print("\t", key, "||| of shape:", weights["default_policy"][key].shape)
            weights_type = key.split(".")[0]
            if weights_type=="core":
                core_curr_list.append(weights["default_policy"][key].flatten())
            if weights_type=="actor":
                actor_curr_list.append(weights["default_policy"][key].flatten())
            if weights_type=="critic":
                critic_curr_list.append(weights["default_policy"][key].flatten())

            if weights_type!= "critic" and  weights_type!= "actor" and weights_type!="core":
                preprocessor_curr_list.append(weights["default_policy"][key].flatten())

            #for all weights totaled together
            curr_weight.append(weights["default_policy"][key].flatten())


            if flag_init==True:
                if weights_type == "core":
                    core_prev_list.append(weights["default_policy"][key].flatten())
                if weights_type == "actor":
                    actor_prev_list.append(weights["default_policy"][key].flatten())
                if weights_type == "critic":
                    critic_prev_list.append(weights["default_policy"][key].flatten())

                if weights_type != "critic" and weights_type != "actor" and weights_type != "core":
                    preprocessor_prev_list.append(weights["default_policy"][key].flatten())

                prev_weights_list.append(weights["default_policy"][key].flatten())


        if flag_init==False:
            curr_total=np.concatenate(curr_weight)
            prev_total=np.concatenate(prev_weights_list)
            weight_diff_l2 = np.linalg.norm(curr_total- prev_total)
            total_l2_norms.append(weight_diff_l2)

            curr_core_total = np.concatenate(core_curr_list)
            prev_core_total = np.concatenate(core_prev_list)
            core_weight_diff_l2 = np.linalg.norm(curr_core_total - prev_core_total)
            core_l2_norms.append(core_weight_diff_l2)

            curr_actor_total = np.concatenate(actor_curr_list)
            prev_actor_total = np.concatenate(actor_prev_list)
            actor_weight_diff_l2 = np.linalg.norm(curr_actor_total - prev_actor_total)
            actor_l2_norms.append(actor_weight_diff_l2)

            curr_preprocessor_total = np.concatenate(preprocessor_curr_list)
            prev_preprocessor_total = np.concatenate(preprocessor_prev_list)
            preprocessor_weight_diff_l2 = np.linalg.norm(curr_preprocessor_total - prev_preprocessor_total)
            preprocessor_l2_norms.append(preprocessor_weight_diff_l2)

            curr_critic_total = np.concatenate(critic_curr_list)
            prev_critic_total = np.concatenate(critic_prev_list)
            critic_weight_diff_l2 = np.linalg.norm(curr_critic_total - prev_critic_total)
            critic_l2_norms.append(critic_weight_diff_l2)


        flag_init = False
        prev_weights_list = curr_weight
        curr_weight = []

        core_prev_list = core_curr_list
        core_curr_list = []

        actor_prev_list = actor_curr_list
        actor_curr_list = []

        critic_prev_list =  critic_curr_list
        critic_curr_list = []

        preprocessor_prev_list= preprocessor_curr_list
        preprocessor_curr_list= []



plt.plot(range(1, len(total_l2_norms)+1), total_l2_norms)
plt.xlabel("Cycle Number")
plt.ylabel("L2 Norm of Weights")
xlabels=sorted([int(x.split("_")[1]) for x in os.listdir(weights_folder_pth)[1:]])
plt.xticks(ticks=range(1, len(total_l2_norms)+1),labels=xlabels)
plt.title("Weight Change L2 Norm")
plt.show()

plt.plot(range(1, len(actor_l2_norms)+1), actor_l2_norms)
plt.xlabel("Cycle Number")
plt.ylabel("L2 Norm of Weights")
xlabels=sorted([int(x.split("_")[1]) for x in os.listdir(weights_folder_pth)[1:]])
plt.xticks(ticks=range(1, len(actor_l2_norms)+1),labels=xlabels)
plt.title("Actor Weight Change L2 Norm")
plt.show()

plt.plot(range(1, len(critic_l2_norms)+1), critic_l2_norms)
plt.xlabel("Cycle Number")
plt.ylabel("L2 Norm of Weights")
xlabels=sorted([int(x.split("_")[1]) for x in os.listdir(weights_folder_pth)[1:]])
plt.xticks(ticks=range(1, len(critic_l2_norms)+1),labels=xlabels)
plt.title("Critic Weight Change L2 Norm")
plt.show()

plt.plot(range(1, len(preprocessor_l2_norms)+1), preprocessor_l2_norms)
plt.xlabel("Cycle Number")
plt.ylabel("L2 Norm of Weights")
xlabels=sorted([int(x.split("_")[1]) for x in os.listdir(weights_folder_pth)[1:]])
plt.xticks(ticks=range(1, len(preprocessor_l2_norms)+1),labels=xlabels)
plt.title("preprocessor Weight Change L2 Norm")
plt.show()

plt.plot(range(1, len(core_l2_norms)+1), core_l2_norms)
plt.xlabel("Cycle Number")
plt.ylabel("L2 Norm of Weights")
xlabels=sorted([int(x.split("_")[1]) for x in os.listdir(weights_folder_pth)[1:]])
plt.xticks(ticks=range(1, len(core_l2_norms)+1),labels=xlabels)
plt.title("core Weight Change L2 Norm")
plt.show()