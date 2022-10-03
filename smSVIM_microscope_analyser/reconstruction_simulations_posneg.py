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
from scipy.fft import rfft, rfftfreq, irfft
# np.random.seed(2)


def gauss(x, mu, s):
    return 1/(2*np.pi*s**2) * np.exp(-(x-mu)**2/(2*s**2))

def rect(x, width = 1, center = 0):
    return np.where(abs(x-center) <=  width/2, 1, 0)


def apply_LPF(matrix, cut_off = 0.1):
    
    lpf_patterns = []
    n, n_obj = matrix.shape
    delta_x = 1

    for i in range(n):
        
        spectrum = rfft(matrix[i,:])
        f = rfftfreq(n_obj, delta_x)
        
        cutoff_idx = abs(f) > cut_off
        spectrum_cut = spectrum.copy()
        spectrum_cut[cutoff_idx] = 0 + 0j
        
        pattern_LPF = irfft(spectrum_cut, n_obj)
        lpf_patterns.append(pattern_LPF)
        
    return np.array(lpf_patterns)


def add_noise(x, noise_type = 'additive', sigma = 1):
    
    if noise_type == 'additive':
        
        noise = np.random.normal(0,sigma, x.shape)
        return x + noise
    
    elif noise_type == ' multiplicatavie':
        
        noise = np.random.normal(1,sigma, x.shape)
        return np.multiply(x , noise)


class Inversion_Simulation:
    
    def __init__(self, n = 16, repeat = 64):
    
        self.n = 16
        self.repeat = 64
        self.n_obj = self.n*self.repeat
    
    
    
    def create_measurement_matrix(self, LPF = False, LPF_cut = 0.03, show = True):
    
        
    
        self.x_pattern = np.linspace(-self.n_obj/2, self.n_obj/2, self.n_obj)
        cont = 100*gauss(self.x_pattern, 0, self.n_obj/3)
        
        pattern_offsett = np.tile(cont, (self.n,1))
        
        self.M = t_6090.create_hadamard_matrix(self.n, 'walsh')
        self.M_ideal = np.repeat(self.M, self.repeat).reshape([self.n, self.n_obj])
        
        
        self.dir_measure_M = np.repeat(self.M, self.repeat).reshape([self.n, self.n_obj])
        self.dir_measure_M[self.dir_measure_M<0] = 0
        self.dir_measure_M = np.multiply(self.dir_measure_M, pattern_offsett) + pattern_offsett
        
        self.dir_measure_M *= 1/np.max(self.dir_measure_M)
        
        
        
        self.dir_measure_M_neg = np.repeat(self.M, self.repeat).reshape([self.n, self.n_obj])
        self.dir_measure_M_neg[self.dir_measure_M_neg<0] = 0
        self.dir_measure_M_neg = np.logical_not(self.dir_measure_M_neg)
        self.dir_measure_M_neg = np.multiply(self.dir_measure_M_neg, pattern_offsett) + pattern_offsett
        self.dir_measure_M_neg *= 1/np.max(self.dir_measure_M_neg)
        
        
        if LPF:
            label = LPF_cut
            self.dir_measure_M = apply_LPF(self.dir_measure_M, cut_off = LPF_cut)
            self.dir_measure_M_neg  = apply_LPF(self.dir_measure_M_neg , cut_off = LPF_cut)
        else:
            label = None
        
        self.posneg_dir_M = self.dir_measure_M - self.dir_measure_M_neg
        
        if show:
            
            # Pos
            fig= plt.figure(figsize = [5,3])
            fig.suptitle(f'Pos matrix (LPF cut = {label})')
            ax = fig.add_subplot(111)
            xy = ax.imshow(self.dir_measure_M, aspect = self.repeat, interpolation = 'none')
            fig.colorbar(xy, ax = ax, format='%.2f')
            
            fig= plt.figure(figsize = [5,3])
            fig.suptitle('Pos last pattern')
            ax = fig.add_subplot(111)
            xy = ax.plot(self.dir_measure_M[-1,:])
            
            # Posneg
            fig= plt.figure(figsize = [5,3])
            fig.suptitle(f'Posneg matrix (LPF cut = {label})')
            ax = fig.add_subplot(111)
            xy = ax.imshow(self.posneg_dir_M, aspect = self.repeat, interpolation = 'none')
            fig.colorbar(xy, ax = ax, format='%.2f')
            
            fig= plt.figure(figsize = [5,3])
            fig.suptitle('Posneg last pattern')
            ax = fig.add_subplot(111)
            xy = ax.plot(self.posneg_dir_M[-1,:])
        
    
    def create_object(self, delta_int = 0.8, delta_z = 0.2, posneg_2n = False, show = True):
        
        self.posneg_2n = posneg_2n
        n_time_points = int((1 + posneg_2n)*self.n)
        self.n_time_points = n_time_points
        
        self.x_obj = np.linspace(0,self.n, self.n_obj)
        
        self.delta_int = delta_int #additional fraction of peak intensity
        self.delta_z = delta_z #percentage of z range
        
        
        
        changing_intensity = np.linspace(30, 30*(1 + self.delta_int), n_time_points)
        changing_position = np.linspace(10, 10 + (self.n*self.delta_z), n_time_points)
        
        self.obj_moving = np.zeros([self.n_obj, n_time_points])
        
        cmap = matplotlib.cm.get_cmap('viridis')
        gradient = np.linspace(0,1, n_time_points)
        gradient_for_im = np.vstack((gradient, gradient)).transpose()
        
        if show:
            fig, axs = plt.subplots(1, 2, tight_layout = True, gridspec_kw={'width_ratios': [20, 1]}, figsize = [5,3])
            fig.suptitle('Unknown fluorescence distribution in ($\\barx$,$\\bary$)')
            
            axs[0].set_xlabel('z (px)')
            axs[0].text(2.5,60, f'$\\Delta I_1$ = {self.delta_int}')
            axs[0].text(10,75, f' $\\Delta z_2$ = {self.delta_z}')
            axs[0].set_ylim([-4, 85])
            
            axs[1].imshow(gradient_for_im, aspect='auto', cmap=cmap)
            axs[1].tick_params(axis = 'x',      # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)
            plt.yticks(ticks = [0, int(n_time_points -1)], labels = ['1', str(n_time_points)])
            
            axs[1].set_ylabel('frame number')
        
        for i in np.linspace(0,n_time_points -1, n_time_points, dtype = int):
            
            
            profile = changing_intensity[i] * rect(self.x_obj, 5, 4) + \
                      70 * rect(self.x_obj, 2, changing_position[i])
                    
            self.obj_moving[:,i] = profile
            
            if i%2 == 0 and show:
                axs[0].plot(self.x_obj, profile, color = cmap(gradient[i]))
            
        
    
    def perform_measurement(self, sigma = 1e3, noise_type = 'additive', show = True):
        
        self.sigma = sigma
        
        if not self.posneg_2n:
            # pos and posneg have the same ammount of frames, not realistic but usefull to compare inversione strenght
        
            self.pos_raw = add_noise( (self.dir_measure_M @ self.obj_moving).diagonal().reshape([self.n,1]),
                                sigma = sigma, noise_type = noise_type )
            
            neg_raw = add_noise( (self.dir_measure_M_neg @ self.obj_moving).diagonal().reshape([self.n,1]),
                                sigma = sigma, noise_type = noise_type)
            
            self.posneg_raw = self.pos_raw - neg_raw
            
        else:
            # same procedure we use in the lab
            self.pos_raw = add_noise( (self.dir_measure_M @ self.obj_moving[:,:self.n]).diagonal().reshape([self.n,1]),
                                sigma = sigma, noise_type = noise_type )
            
            pos_raw_2n = add_noise( (self.dir_measure_M @ self.obj_moving[:,::2]).diagonal().reshape([self.n,1]),
                                sigma = sigma, noise_type = noise_type )
            
            neg_raw_2n = add_noise( (self.dir_measure_M_neg @ self.obj_moving[:,1::2]).diagonal().reshape([self.n,1]),
                                sigma = sigma, noise_type = noise_type)
        
            self.posneg_raw = pos_raw_2n - neg_raw_2n
            
        # perfect matrix but stil moving sample
        self.ideal_raw = (self.M_ideal @ self.obj_moving[:,:self.n]).diagonal()
        
        self.make_posneg_raw = 2*(self.pos_raw - np.mean(self.pos_raw[7:]))
    
        if show:
            # SPECTRA
            
            fig = plt.figure(figsize = [5,3])
            ax = fig.add_subplot(111)
            fig.suptitle(f'Walsh-Hadamard spectrum (noise $\sigma$ = {self.sigma:.0f})')
            ax.set_xlabel('Walsh pattern number')
            ax.set_ylabel('Intensity at cemra pixel')
            
            ax.plot(self.ideal_raw,'--', label = 'Ideal Walsh spectrum', color = 'C2')
            ax.plot(self.pos_raw, label = 'Pos measured', color = 'C1')
            ax.plot(self.posneg_raw, '-o', label = 'Posneg measured', color = 'C0')
            ax.text(12,-3e4, f'$\\Delta I_1$ = {self.delta_int}\n$\\Delta z_2$ = {self.delta_z}')

            
            # make_posneg_raw = 2*pos_raw - pos_raw[0]
            # ax.plot(make_posneg_raw, label = 'make_posneg_raw')
            
            
            ax.plot(self.make_posneg_raw,'-', label = 'Pos measured + make_posneg', color = 'C3')
            ax.set_ylim([-35e3, 20e3])
            ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
            ax.legend()
    
    def invert(self, show = True):
    
        # I use the ideal matrix to invert the raw data
        # self.inv_posneg = 1/self.n*self.M @ self.posneg_raw
        self.inv_posneg_lsqr,_,_,_ = lstsq(self.M, self.posneg_raw, rcond = None)
        
        # Here I use the ideal matrix with only positive values
        self.M_pos = self.M.copy()
        self.M_pos[self.M_pos<0] = 0
        self.inv_pos ,_,_,_ = lstsq(self.M_pos, self.pos_raw, rcond = None)
        
        # Ideal matrix since I apply make_posneg
        # self.inv_make_posneg = 1/self.n*self.M @ self.make_posneg_raw
        self.inv_make_posneg_lsqr ,_,_,_ = lstsq(self.M, self.make_posneg_raw, rcond = None)
        
        
        # I remove the CC
        self.pos_raw_no_CC = self.pos_raw[1:]
        self.M_pos_no_CC =self. M_pos[1:,:]
        self.inv_pos_no_CC ,_,_,_ = lstsq(self.M_pos_no_CC, self.pos_raw_no_CC, rcond = None)
        
        
        
        if show:
            fig = plt.figure(figsize = [5,3])
            ax = fig.add_subplot(111)
            
            fig.suptitle(f'Inverted profiles (noise $\sigma$ = {self.sigma:.0f})')
            ax.set_xlabel('z (px)')
            ax.set_ylabel('gray scale intensity')
            ax.plot(self.x_obj - 0.5*np.ones(self.x_obj.shape), 30*self.obj_moving[:,0], 
                    label = 'Scaled object at t = 0', color = 'C0')
            ax.plot(self.x_obj - 0.5*np.ones(self.x_obj.shape), 30*self.obj_moving[:,-1],'--', linewidth = 0.7, 
                    label = f'Scaled object at t = {self.n_time_points}', color = 'C0')
            
            # ax.plot(inv_posneg, '-o', label = 'inv posneg')
            ax.plot(self.inv_posneg_lsqr, '-', linewidth = 4, color = 'C1',
                    label = f'posneg inverted ({self.n_time_points} frames)')
            
            ax.plot(self.inv_pos, '-x', markersize = 6, color = 'C2',
                    label = f'pos inverted ({self.n} frames)')
            # ax.plot(inv_make_posneg, '-o', label = 'inv fake posneg')
            ax.plot(self.inv_make_posneg_lsqr, '-o', markersize = 4, color = 'C3',
                    label = 'pos + make_posneg inverted')
            
            ax.plot(self.inv_pos_no_CC, '--', color = 'C4', linewidth = 0.6, label = 'pos + no CC inverted')
            
            
            
            ax.set_ylim([-2e3, 12e3])
            ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
            ax.text(1,8e3, f'$\\Delta I_1$ = {self.delta_int}\n$\\Delta z_2$ = {self.delta_z}')
            ax.legend()
        
        
    
    def light_sheet(self, sigma = 1e3, noise_type = 'additive', show = True):
        # Perfect light sheet, for example having the sample moving through the light sheet
        
        self.M_ls = np.repeat(np.eye(self.n), self.repeat).reshape(self.n, self.n_obj)
        # print(M_ls)
        
        
        self.ls_raw = add_noise((self.M_ls @ self.obj_moving[:,:self.n]).diagonal().reshape([self.n,1]),
                                noise_type = noise_type, sigma=sigma)
        
        if show:
        
            fig = plt.figure(figsize = [5,3])
            ax = fig.add_subplot(111)
            
            fig.suptitle(f'Light Sheet profile (noise $\sigma$ = {sigma:.0f})')
            ax.set_xlabel('z (px)')
            ax.set_ylabel('gray scale intensity')
            
            ax.plot(self.x_obj - 0.5*np.ones(self.x_obj.shape), 65*self.obj_moving[:,0],
                    color = 'C0', label = 'Scaled object at t = 0')
            ax.plot(self.x_obj - 0.5*np.ones(self.x_obj.shape), 65*self.obj_moving[:,-1],
                    '--', linewidth = 0.7, label = f'Scaled object at t = {self.n}', color = 'C0')
            
            ax.plot(self.ls_raw, '-o', color = 'C1', label = ' Light Sheet')
            
            ax.text(0,6e3, f'$\\Delta I_1$ = {self.delta_int}\n$\\Delta z_2$ = {self.delta_z}')
            ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
            ax.legend()
            ax.set_ylim([-300, 8e3])
            
            
        
        
        
        
if __name__ == '__main__':
    
    simul = Inversion_Simulation()
    
    # noise
    sigma = 0
    noise_type = 'additive' # 'additive' or 'multiplicative'

    # create measurement and unknown object
    simul.create_measurement_matrix(LPF = True, LPF_cut=0.03, show = True)   
    simul.create_object(delta_int = 0, delta_z =0, posneg_2n = True, show = True)
    
    # calculate walsh spectrum and invert
    simul.perform_measurement(sigma = sigma, noise_type = noise_type, show = True)
    simul.invert(show = True)
    
    #%% Light Sheet Microscopy
    
    simul.light_sheet(sigma = sigma, noise_type = noise_type, show = True)
    
