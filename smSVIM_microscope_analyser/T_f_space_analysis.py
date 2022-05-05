#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:37:23 2022

@author: marcovitali
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import transform_6090 as t_6090


disp_freqs = np.array([0, 0.5, 1.0, 1.5037593984962405, 2.0, 2.5, 3.0303030303030303, 3.508771929824561, 4.0, 4.545454545454546, 5.0, 5.555555555555555, 6.0606060606060606, 6.666666666666667, 7.142857142857143, 7.6923076923076925, 8.0, 8.695652173913043, 9.090909090909092, 9.523809523809524, 10.0, 10.526315789473685, 11.11111111111111, 11.764705882352942, 12.5, 12.5, 13.333333333333334, 14.285714285714286, 14.285714285714286, 15.384615384615385, 15.384615384615385, 16.666666666666668, 16.666666666666668, 16.666666666666668, 18.181818181818183, 18.181818181818183, 18.181818181818183, 20.0, 20.0, 20.0, 20.0, 22.22222222222222, 22.22222222222222, 22.22222222222222, 22.22222222222222, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 33.333333333333336, 33.333333333333336, 33.333333333333336])

mask = np.append(np.diff(disp_freqs)!= 0, True)
disp_freqs = np.array(disp_freqs)[mask]

print(f'\nWe have {len(disp_freqs)} out of {len(mask)} independent measures\n(difference: {len(mask)- len(disp_freqs)})\n\n')

temp = disp_freqs.copy()

temp[0] = 0.25
periods = np.reciprocal(temp)*200

print('Displayed\n  T(px)     f')
np.set_printoptions(suppress=True, precision = 2)
print(np.vstack((periods, disp_freqs)).T)

fig, ax = plt.subplots(1,1)

ax.plot(disp_freqs, periods, 'o', markersize = 1, label = 'DCT approx.')
ax.set_xlabel('Frequency')
ax.set_ylabel('Period')

# fig2, ax2 = plt.subplots(1,1)
# ax2.plot(disp_freqs, periods, 'o-')
# ax2.set_yscale('log')
# ax2.set_xlabel('f')
# ax2.set_ylabel('T')

print('\n\nTo add\n  T(px)     f')
per_to_add = np.array([24.0, 27.0, 29.0, 31, 32, 34 , 35, 37, 38, 39])
# per_to_add = np.array([24.0, 27.0, 29.0, 31, 32, 34 , 35, 37, 38, 39 , 42, 46, 48, 52, 55, 59, 61, 64, 68 ])
freq_to_add = 200*np.reciprocal(per_to_add)
print(np.vstack((per_to_add, freq_to_add)).T)
print(f'\nWe should add {len(mask)- len(disp_freqs)- len(freq_to_add)} more frequencies')


ax.plot(freq_to_add, per_to_add, 'd', color = 'C1', markersize = 1, label= 'Frequencies we could add')
ax.legend()

new_freq = np.append(disp_freqs, freq_to_add)
new_freq = np.sort(new_freq)
# print(new_freq)



tras = t_6090.dct_6090(new_freq)
tras.create_space()

tras.create_matrix_cos()
tras.compute_inverse() # the inverse matrix is now in tras.inv_matrix
tras.compute_pinv()





