#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:51:55 2022

@author: marcovitali
"""
import numpy as np
import matplotlib.pyplot as plt

#cos
maxs = np.array([ 325,  1303,3587, 5986, 8378])
ws = np.array([ 40, 190, 420,750, 1000])


#sq
# maxs = np.array([347, 918, 5871])
# ws = np.array([71, 300, 1200])


fig, ax = plt.subplots(1,1)

ax.plot(maxs, ws)

