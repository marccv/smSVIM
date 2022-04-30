#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:21:33 2022

@author: marcovitali
"""
import numpy as np



# a = np.array([[[1, 3],[5,7]],[[2,4],[6,8]]])
# b = np.array([[0,1],[1,0]])

# print('b\n', b)
# print('\na\n', a)

# c = np.tensordot(b,a, axes=([1],[0]))
# print('\n\ntensordot c\n', c)

# # print(c.clip(min = 0))

# # print((c/2).astype(int))

# e = 3*a%c

# print(e)



# a = np.array([1,2,3,4,4,5,8,8,8, 9, 9])
# mask = np.append(np.logical_not((np.diff(a))== 0), True)
# mask = np.append(mask, True)

# print(a)
# print(mask)

# print(a[mask])


# fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)



# a = np.array([[[10,11],[12,13]],[[14,15],[16,17]] , [[12,11],[12,116]],[[14,15],[16,17]] ])
# print('\na\n', a)

# num_frames = a.shape[0]
# print(f'num_frames = {num_frames}')

# pos = a[np.linspace(0, num_frames -2, int(num_frames/2), dtype = 'int'), :, :]
# neg = a[np.linspace(1, num_frames -1, int(num_frames/2), dtype = 'int'), :, :]

# print('\npos\n', pos)
# print('\nneg\n', neg)


# b = pos - neg
# print('\nb\n', b)


import scipy
import scipy.misc
import matplotlib.pyplot as plt

f = scipy.misc.face()

fig1 , ax1 = plt.subplots()

ax1.imshow(f)
