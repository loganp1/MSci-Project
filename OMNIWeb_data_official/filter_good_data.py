# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:43:48 2023

@author: Ned
"""

import numpy as np
import matplotlib.pyplot as plt
import time
t1=time.time()
#periods=np.loadtxt("goodPeriods.txt")
#periods=np.append(periods,[])
def modify_list(input_list, min_difference, max_difference):
    new_list = []

    for sublist in input_list:
        diff = sublist[1] - sublist[0]

        if diff >= min_difference:  # Only proceed if the difference meets the threshold
            if diff > max_difference:
                num_splits = int(diff // max_difference)  # Convert to integer
                remainder = diff % max_difference
                split_diff = (diff - remainder) // num_splits

                for i in range(num_splits):
                    start = sublist[0] + (split_diff * i)
                    end = sublist[0] + (split_diff * (i + 1))
                    if end - start >= min_difference:  # Check the difference after split
                        new_list.append([start, end])

                if remainder > 0 and sublist[1] - (sublist[0] + (split_diff * num_splits)) >= min_difference:
                    start = sublist[0] + (split_diff * num_splits)
                    end = sublist[1]
                    new_list.append([start, end])
            else:
                new_list.append(sublist)

    return new_list



# test=modify_list(periods,90*60,180*60)
# differences = [inner_list[1] - inner_list[0] for inner_list in test]
# np.savetxt("modPeriods.txt",test)
# plt.hist(differences)
# print(time.time()-t1)