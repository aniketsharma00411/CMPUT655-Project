import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

weights_folder_pth="weights"


flag_init = True
l2_norms=[]
prev_weights_list = []
curr_weight=[]
for cycle_num in range(len(os.listdir(weights_folder_pth))):

    print("For Cycle :", cycle_num)

    with open(f"{weights_folder_pth}/weights_{cycle_num}_test.pkl", "rb") as f:
        weights = pickle.load(f)

        print("\t Types of weights found in the pickle file of numbers: ", len(weights["default_policy"].keys()))
        for key in weights["default_policy"].keys():
            curr_weight.append(weights["default_policy"][key].flatten())
            print("\t",key,"||| of shape:",weights["default_policy"][key].shape)
            if flag_init==True:
                prev_weights_list.append(weights["default_policy"][key].flatten())
                #print("\t Previous weight shape",prev_weight.shape)
                # Calculate the L2 norm of the weights

        # print(prev_weights_list)

        if flag_init==False:
            curr_total=np.concatenate(curr_weight)
            prev_total=np.concatenate(prev_weights_list)

            weight_diff_l2 = np.linalg.norm(curr_total- prev_total)
            l2_norms.append(weight_diff_l2)
        flag_init = False
        prev_weights_list = curr_weight
        curr_weight = []


plt.plot(range(1, len(l2_norms)+1), l2_norms)
plt.xlabel("Cycle Number")
plt.ylabel("L2 Norm of Weights")
xlabels=sorted([int(x.split("_")[1]) for x in os.listdir(weights_folder_pth)[1:]])
plt.xticks(ticks=range(1, len(l2_norms)+1),labels=xlabels)
plt.title("Weight Change L2 Norm")
plt.show()
