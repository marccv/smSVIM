#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 19:30:33 2022

@author: marcovitali
"""

import numpy as np
import scipy.fftpack as sp_fft
import matplotlib.pyplot as plt



Y = np.zeros([1,100])
Y[0,1] = 1
# Y[0,3] = 0.5

y = sp_fft.idct(Y, norm = 'ortho')



fig1=plt.figure(num=1, figsize=(5,4))
fig1.clf()

ax1=fig1.add_subplot(111)


# fig1.text (0.2 ,0.92,'a)  $\qquad$Signal v. time', fontsize = 16)
ax1.plot( y[0,:])
# ax1.set_xlabel('Time ($\mu s$)', fontsize = 14)
# ax1.set_ylabel('Voltage (V)', fontsize = 14);




# %%

x = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,5,9,10,9,5,3,1,0,0,0,0,0,])

X = sp_fft.dct(x,norm = 'ortho', type = 2)

x_prime = sp_fft.idct(X, norm = 'ortho', type = 2)

fig1=plt.figure(num=1, figsize=(5,4))
fig1.clf()

ax1=fig1.add_subplot(121)

ax1.plot(x, 'o')
ax1.plot(x_prime, 'x',color = 'C1')

ax2 = fig1.add_subplot(122)
ax2.plot(X, '.')

