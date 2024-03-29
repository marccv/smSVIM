#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:01:42 2022

@author: marcovitali

"""


import numpy as np
import transform_6090 as t_6090
import matplotlib.pyplot as plt
import matplotlib
from numpy.linalg import lstsq
# np.random.seed(2)

def gauss(x, mu, s):
    return 1/(2*np.pi*s**2) * np.exp(-(x-mu)**2/(2*s**2))


n = 16
repeat = 64
n_obj = n*repeat


x_pattern = np.linspace(-n_obj/2, n_obj/2, n_obj)
cont = 100*gauss(x_pattern, 0, n_obj/3)

pattern_offsett = np.tile(cont, (n,1))

M = t_6090.create_hadamard_matrix(n, 'walsh')


dir_measure_M = np.repeat(M, repeat).reshape([n,n_obj])
dir_measure_M[dir_measure_M<0] = 0
dir_measure_M = np.multiply(dir_measure_M, pattern_offsett) + pattern_offsett

dir_measure_M *= 1/np.max(dir_measure_M)



dir_measure_M_neg = np.repeat(M, repeat).reshape([n,n_obj])
dir_measure_M_neg[dir_measure_M_neg<0] = 0
dir_measure_M_neg = np.logical_not(dir_measure_M_neg)
dir_measure_M_neg = np.multiply(dir_measure_M_neg, pattern_offsett) + pattern_offsett
dir_measure_M_neg *= 1/np.max(dir_measure_M_neg)


posneg_dir_M = dir_measure_M - dir_measure_M_neg



# Pos
fig= plt.figure(figsize = [5,3])
fig.suptitle('measurement pos matrix')
ax = fig.add_subplot(111)
xy = ax.imshow(dir_measure_M, aspect = repeat, interpolation = 'none')
cbar = fig.colorbar(xy, ax = ax, format='%.2f')

fig= plt.figure(figsize = [5,3])
fig.suptitle('measurement pos last pattern')
ax = fig.add_subplot(111)
xy = ax.plot(dir_measure_M[-1,:])

# Posneg
fig= plt.figure(figsize = [5,3])
fig.suptitle('measurement posneg matrix')
ax = fig.add_subplot(111)
xy = ax.imshow(posneg_dir_M, aspect = repeat, interpolation = 'none')
cbar = fig.colorbar(xy, ax = ax, format='%.2f')

fig= plt.figure(figsize = [5,3])
fig.suptitle('measurement posneg last pattern')
ax = fig.add_subplot(111)
xy = ax.plot(posneg_dir_M[-1,:])



#%% OBJECT



def rect(x, width = 1, center = 0):
    return np.where(abs(x-center) <=  width/2, 1, 0)


x = np.linspace(0,n, n_obj)

delta_int = 0.8 #additional fraction of peak intensity
# delta_int = 0

delta_z = 0.2 #percentage of z range
# delta_z = 0

changing_intensity = np.linspace(30, 30*(1 + delta_int), n)
changing_position = np.linspace(10, 10 + (n*delta_z), n)

obj_moving = np.zeros([n_obj,n])

cmap = matplotlib.cm.get_cmap('viridis')
gradient = np.linspace(0,1, n)
gradient_for_im = np.vstack((gradient, gradient)).transpose()

fig, axs = plt.subplots(1, 2, tight_layout = True, gridspec_kw={'width_ratios': [20, 1]}, figsize = [5,3])
fig.suptitle('Unknown fluorescence distribution in ($\\barx$,$\\bary$)')

axs[0].set_xlabel('z (px)')
axs[0].text(2.5,60, f'$\\Delta I_1$ = {delta_int}')
axs[0].text(10,75, f' $\\Delta z_2$ = {delta_z}')
axs[0].set_ylim([-4, 85])

fig = axs[1].imshow(gradient_for_im, aspect='auto', cmap=cmap)
axs[1].tick_params(axis = 'x',      # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.yticks(ticks = [0, int(n-1)], labels = ['1', str(n)])

axs[1].set_ylabel('frame number')

for i in range(n):
    
    
    profile = changing_intensity[i] * rect(x, 5, 4) + \
              70 * rect(x, 2, changing_position[i])
            
    obj_moving[:,i] = profile

    axs[0].plot(x, profile, color = cmap(gradient[i]))
    


#%%

#-----------------
# Additive noise
#-----------------

# sigma = 1e3
sigma = 0
det_noise_pos = np.random.normal(0,sigma, (n,1))
det_noise_neg = np.random.normal(0,sigma, (n,1))

pos_raw = ((dir_measure_M@obj_moving).diagonal()).reshape([n,1]) + det_noise_pos
neg_raw = ((dir_measure_M_neg@obj_moving).diagonal()).reshape([n,1]) + det_noise_neg

posneg_raw = pos_raw - neg_raw

#----------------------
# multiplicative noise
#----------------------

# sigma = 0.05
# det_noise_pos = np.random.normal(1,sigma, (n,1))
# det_noise_neg = np.random.normal(1,sigma, (n,1))

# pos_raw = np.multiply( ((dir_measure_M@obj_moving).diagonal()).reshape([n,1]), det_noise_pos)
# neg_raw = np.multiply( ((dir_measure_M_neg@obj_moving).diagonal()).reshape([n,1]) , det_noise_neg)

# posneg_raw = pos_raw - neg_raw



M_ideal = np.repeat(M, repeat).reshape([n,n_obj])
ideal_raw = (M_ideal@obj_moving).diagonal()


# SPECTRA

fig = plt.figure(figsize = [5,3])
ax = fig.add_subplot(111)
fig.suptitle(f'Walsh-Hadamard spectrum (noise $\sigma$ = {sigma:.0f})')
ax.set_xlabel('Walsh pattern number')
ax.set_ylabel('Intensity at cemra pixel')

ax.plot(ideal_raw,'--', label = 'Ideal Walsh spectrum', color = 'C2')
ax.plot(pos_raw, label = 'Pos measured', color = 'C1')
ax.plot(posneg_raw, '-o', label = 'Posneg measured', color = 'C0')
ax.text(0,-3e4, f'$\\Delta I_1$ = {delta_int}\n$\\Delta z_2$ = {delta_z}')




# make_posneg_raw = 2*pos_raw - pos_raw[0]
# ax.plot(make_posneg_raw, label = 'make_posneg_raw')


make_posneg_raw = 2*(pos_raw - np.mean(pos_raw[7:]))
ax.plot(make_posneg_raw,'-', label = 'Pos measured + make_posneg', color = 'C3')
ax.set_ylim([-35e3, 20e3])
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
ax.legend()

#%%

# I use the ideal matrix to invert the raw data
inv_posneg = 1/n*M @ posneg_raw
inv_posneg_lsqr,_,_,_ = lstsq(M, posneg_raw, rcond = None)

# Here I use the ideal matrix with only positive values
M_pos = M.copy()
M_pos[M_pos<0] = 0
inv_pos ,_,_,_ = lstsq(M_pos, pos_raw, rcond = None)

# Ideal matrix since I apply make_posneg
inv_make_posneg = 1/n*M @ make_posneg_raw
inv_make_posneg_lsqr ,_,_,_ = lstsq(M, make_posneg_raw, rcond = None)


# I remove the CC
pos_raw_no_CC = pos_raw[1:]
M_pos_no_CC = M_pos[1:,:]
inv_pos_no_CC ,_,_,_ = lstsq(M_pos_no_CC, pos_raw_no_CC, rcond = None)


fig = plt.figure(figsize = [5,3])
ax = fig.add_subplot(111)

fig.suptitle(f'Inverted profiles (noise $\sigma$ = {sigma:.0f})')
ax.set_xlabel('z (px)')
ax.set_ylabel('gray scale intensity')
ax.plot(x - 0.5*np.ones(x.shape), 30*obj_moving[:,0], label = 'Scaled object at t = 0')

# ax.plot(inv_posneg, '-o', label = 'inv posneg')
ax.plot( inv_posneg_lsqr, '-', linewidth = 4,label = 'posneg inverted')

ax.plot(inv_pos, '-x', markersize = 6, label = 'pos inverted')
# ax.plot(inv_make_posneg, '-o', label = 'inv fake posneg')
ax.plot(inv_make_posneg_lsqr, '-o', markersize = 4,label = 'pos + make_posneg inverted')

ax.plot(inv_pos_no_CC, '--', label = 'pos + no CC inverted')

ax.plot(x - 0.5*np.ones(x.shape), 30*obj_moving[:,-1],'--', linewidth = 0.7, label = f'Scaled object at t = {n-1}', color = 'C0')

ax.set_ylim([-2e3, 12e3])
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
ax.text(1,8e3, f'$\\Delta I_1$ = {delta_int}\n$\\Delta z_2$ = {delta_z}')
ax.legend()




#%% LIGHT SHEET
# Perfect light sheet, for example having the sample moving through the light sheet

M_ls = np.repeat(np.eye(n), repeat).reshape(n, n_obj)
# print(M_ls)


#----------------------
# additive noise
#----------------------

sigma_ls = sigma # change here to make them different
det_noise_ls = np.random.normal(0,sigma_ls, (n,1))
ls_raw = ((M_ls@obj_moving).diagonal()).reshape([n,1]) + det_noise_ls


#----------------------
# multiplicative noise
#----------------------

# sigma_ls = sigma
# det_noise_ls = np.random.normal(1,sigma_ls, (n,1))
# ls_raw = np.multiply( ((M_ls@obj_moving).diagonal()).reshape([n,1]), det_noise_ls)




fig = plt.figure(figsize = [5,3])
ax = fig.add_subplot(111)

fig.suptitle(f'Light Sheet profile (noise $\sigma$ = {sigma_ls:.0f})')
ax.set_xlabel('z (px)')
ax.set_ylabel('gray scale intensity')

ax.plot(x - 0.5*np.ones(x.shape), 65*obj_moving[:,0], color = 'C0', label = 'Scaled object at t = 0')
ax.plot(x - 0.5*np.ones(x.shape), 65*obj_moving[:,-1],'--', linewidth = 0.7, label = f'Scaled object at t = {n-1}', color = 'C0')

ax.plot(ls_raw, '-o', color = 'C1', label = ' Light Sheet')

ax.text(0,6e3, f'$\\Delta I_1$ = {delta_int}\n$\\Delta z_2$ = {delta_z}')
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
ax.legend()
ax.set_ylim([-300, 8e3])

