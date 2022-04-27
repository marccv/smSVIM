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



a = np.array([1,2,3,4,4,5,8,8,8, 9, 9])
mask = np.append(np.logical_not((np.diff(a))== 0), True)
# mask = np.append(mask, True)

print(a)
# print(mask)

print(a[mask])


# fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)