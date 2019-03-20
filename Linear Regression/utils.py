# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:24:49 2019

@author: yanming
"""
import numpy as np


# get random linear data y = k*x
def get_linear_data(num):
    if num <= 1:
        print("The length of data should be more than 1!")
        pass
    else:
        x = np.arange(1,num+1)  # get axis x
        rand_k = 5 + (5-3)*np.random.random()  # get random k in range [1,3]
        rand_noise = np.random.uniform(-1,1,num)*(num/10)  # get random noise
        y = rand_k*x + rand_noise
        return x, y, rand_k

