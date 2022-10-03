#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:28:50 2022

@author: marcovitali
"""


import numpy as np
import transform_6090 as t_6090
import matplotlib.pyplot as plt
# from numpy.linalg import lstsq

from scipy.fft import rfft, rfftfreq, irfft

# np.random.seed(2)

def gauss(x, mu, s):
    return 1/(2*np.pi*s**2) * np.exp(-(x-mu)**2/(2*s**2))


n = 16
repeat = 64
n_obj = n*repeat


x_pattern = np.linspace(-n_obj/2, n_obj/2, n_obj, endpoint = False)

cont = 100*gauss(x_pattern, 0, n_obj/3)

pattern_offsett = np.tile(cont, (n,1))

M = t_6090.create_hadamard_matrix(n, 'walsh')


dir_measure_M = np.repeat(M, repeat).reshape([n,n_obj])
dir_measure_M[dir_measure_M<0] = 0
dir_measure_M = np.multiply(dir_measure_M, pattern_offsett) + pattern_offsett

dir_measure_M *= 1/np.max(dir_measure_M)

lpf_patterns = []

for i in range(16):
        
    
    pattern = dir_measure_M[i,:]
    
    
    spectrum = rfft(pattern)
    f = rfftfreq(n_obj, x_pattern[1]-x_pattern[0])
    
    
    p_s = np.abs(spectrum)
    
    # fig = plt.figure(figsize = [5,3])
    # fig.suptitle(f'Pattern {i +1} spectrum ^2')
    # ax = fig.add_subplot(111)
    
    # ax.plot(f, 2/n_obj*p_s)
    
    cutoff_idx = abs(f) > 0.03
    spectrum_cut = spectrum.copy()
    spectrum_cut[cutoff_idx] = 0 + 0j
    
    pattern_LPF = irfft(spectrum_cut, n_obj)
    lpf_patterns.append(pattern_LPF)
    
    # if i == 0:
    #     scale = 1/max(pattern_LPF)
        
    # pattern_LPF *= scale
    
    
    # fig = plt.figure(figsize = [5,3])
    # fig.suptitle(f'Pattern {i +1}')
    # ax = fig.add_subplot(111)
    
    # ax.plot(x_pattern, pattern,'--', linewidth = 0.5, label = 'pattern without LPF')
    # ax.plot(x_pattern, pattern_LPF, label = 'LPF')
    # ax.legend()
    # ax.set_ylim([0, 1.5])

lpf_patterns = np.array(lpf_patterns)


# Pos
fig= plt.figure(figsize = [5,3])
fig.suptitle('measurement pos matrix')
ax = fig.add_subplot(111)
xy = ax.imshow(lpf_patterns, aspect = repeat, interpolation = 'none')
cbar = fig.colorbar(xy, ax = ax, format='%.2f')

fig= plt.figure(figsize = [5,3])
fig.suptitle('measurement pos last pattern')
ax = fig.add_subplot(111)
xy = ax.plot(lpf_patterns[-1,:])

