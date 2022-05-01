#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:25:23 2022

@author: marcovitali
"""

import numpy as np
import transform_6090 as t_6090
import scipy.fftpack as sp_fft
from get_h5_data import get_h5_dataset, get_h5_attr
import tifffile as tiff
import time
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 7})

from skimage.restoration import denoise_tv_chambolle

import sys
import pyqtgraph as pg
import qtpy.QtCore
from qtpy.QtWidgets import QApplication



def time_it(method):
    """Fucntion decorator to time a methos""" 
       
    def inner(*args,**kwargs):
        
        start_time = time.time() 
        result = method(*args, **kwargs) 
        end_time = time.time()
        print(f'Execution time for method "{method.__name__}": {end_time-start_time:.6f} s \n') 
        return result        
    return inner





class coherentSVIM_analysis:
    
    name = 'coherentSVIM_analysis'
    
    # disp_freq = np.array([0, 0.5, 1.0, 1.5037593984962405, 2.0, 2.5, 3.0303030303030303, 3.508771929824561, 4.0, 4.545454545454546, 5.0, 5.555555555555555, 6.0606060606060606, 6.666666666666667, 7.142857142857143, 7.6923076923076925, 8.0, 8.695652173913043, 9.090909090909092, 9.523809523809524, 10.0, 10.526315789473685, 11.11111111111111, 11.764705882352942, 12.5, 12.5, 13.333333333333334, 14.285714285714286, 14.285714285714286, 15.384615384615385, 15.384615384615385, 16.666666666666668, 16.666666666666668, 16.666666666666668, 18.181818181818183, 18.181818181818183, 18.181818181818183, 20.0, 20.0, 20.0, 20.0, 22.22222222222222, 22.22222222222222, 22.22222222222222, 22.22222222222222, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 33.333333333333336, 33.333333333333336, 33.333333333333336])
    
    
    
    def __init__(self, fname):
        
        self.filename  = fname
        
    @time_it   
    def load_h5_file(self):
        
        self.imageRaw = get_h5_dataset(self.filename) #TODO read h5 info
        
        #load some settings, if present in the h5 file
        # for key in ['exposure', 'transpose_pattern','ROI_s_z','ROI_s_y']:
        #     val = get_h5_attr(self.filename, key)
        #     print(val)
            
    def show_im_raw(self):
        pg.image(self.imageRaw, title="Raw image")        
               
        #keeps the window open running a QT application
        if sys.flags.interactive != 1 or not hasattr(qtpy.QtCore, 'PYQT_VERSION'):
            QApplication.exec_()
                          
   
        sys.exit ( "End of test")
        
    @time_it    
    def show_im_raw_cc(self):
        
        fig1=plt.figure()
        fig1.clf()
        
        ax1=fig1.add_subplot(111)
        fig1.suptitle('Raw image uniform illumination')
        xy = ax1.imshow(self.imageRaw[0,:,:].transpose(), cmap = 'gray', aspect = 1, interpolation = 'none') 
        ax1.set_xlabel('x (px)')
        ax1.set_ylabel('y (px)')
        cbar = fig1.colorbar(xy, ax = ax1, shrink=1, format='%.0e')
        cbar.ax.set_ylabel('Counts', rotation=270)
    
    @time_it
    def merge_pos_neg(self):
        
        num_frames = self.imageRaw.shape[0]
        
        pos = self.imageRaw[np.linspace(0, num_frames -2, int(num_frames/2), dtype = 'int'), :, :]
        neg = self.imageRaw[np.linspace(1, num_frames -1, int(num_frames/2), dtype = 'int'), :, :]
        
        self.imageRaw = pos - neg
    
    @time_it
    def setROI(self, x_min, y_min, ROIsize, ROIsize_y = None):
        
        '''
        Defines the ROI size
        If ROIsize_y is not provided the ROI becomes a square of side ROIsize
        '''
        
        if ROIsize_y == None:
            ROIsize_y = ROIsize
        
        self.imageRaw = self.imageRaw[:, x_min :x_min +ROIsize, y_min: y_min+ROIsize_y]
    
    @time_it
    def choose_freq(self, N = None):
        
        f_start = get_h5_attr(self.filename, 'f_min')[0]
        f_stop = get_h5_attr(self.filename, 'f_max')[0]
        ROI_s_z = get_h5_attr(self.filename, 'ROI_s_z')[0]
        self.ROI_s_z = ROI_s_z
        
        freqs = np.linspace(f_start, f_stop, int(2*(f_stop - f_start) + 1),dtype = float)
        disp_freqs = [0]
        
        for freq in freqs[1:]:
            period = int(ROI_s_z/freq)
            disp_freqs.append(ROI_s_z/period)
        
        
        self.imageRaw = self.imageRaw[0:N, :, :]
        self.disp_freqs = disp_freqs[0:N]
        
        # to eliminate copies of the same frequency
        # mask = np.append(np.diff(disp_freqs)!= 0, True)
        
        # self.imageRaw = self.imageRaw[mask, :, :]
        # self.disp_freqs = np.array(disp_freqs)[mask]
    
    @time_it    
    def invert(self, base = 'cos'):
        
        self.base = base
        
        self.transform = t_6090.dct_6090(self.disp_freqs)
        self.transform.create_space()
        
        if base == 'cos':
            self.transform.create_matrix_cos()
            self.transform.compute_inverse()
            self.image_inv = np.tensordot(self.transform.inv_matrix ,  self.imageRaw , axes=([1],[0]))
            
        elif base == 'sq':
            self.transform.create_matrix_sq()
            self.transform.compute_inverse()
            self.image_inv = np.tensordot(self.transform.inv_matrix ,  self.imageRaw , axes=([1],[0]))
            
        elif base == 'sp_dct':
            dct_coeff = self.imageRaw
            dct_coeff[0,:,:] *= 1/np.sqrt(2)  # I rescale the cw illumination >> It just shifts the inverted image towards more negative values
            self.image_inv = sp_fft.idct(dct_coeff, type = 2, axis = 0, norm = 'ortho')
        
        self.denoised = False
        self.clipped = False
    
    @time_it        
    def p_invert(self, base = 'cos'):
        
        '''
        Inverts the raw image using the the matrix pseudoinverse with rcond = 10
        '''
        
        self.base = base
        
        self.transform = t_6090.dct_6090(self.disp_freqs)
        self.transform.create_space()
        
        if base == 'cos':
            self.transform.create_matrix_cos()
            self.transform.compute_pinv()
            self.image_inv = np.tensordot(self.transform.pinv_matrix ,  self.imageRaw , axes=([1],[0]))
            
        elif base == 'sq':
            self.transform.create_matrix_sq()
            self.transform.compute_pinv()
            self.image_inv = np.tensordot(self.transform.pinv_matrix ,  self.imageRaw , axes=([1],[0]))
        
        self.denoised = False
        self.clipped = False
    
    @time_it
    def denoise(self, weight = 1000):
        
        shape = self.image_inv.shape
        self.denoise_w = 'linear'
        if self.base == 'cos':
            ratio = 0.1193
        elif self.base == 'sq':
            ratio = 0.20439
        
        for i in range(shape[1]):
            for j in range(shape[2]):
                
                weight = max(self.image_inv[:,i,j]) * ratio
                self.image_inv[:,i,j] = denoise_tv_chambolle(self.image_inv[:,i,j], weight)
        
        self.denoised = True
    
    @time_it
    def cut_negatives(self):
        
        self.image_inv = self.image_inv.clip(min = 0)
        self.clipped = True
    
    def show_inverted(self):
        pg.image(self.image_inv, title= f"Inverted image (base: {self.base})")        
               
        #keeps the window open running a QT application
        if sys.flags.interactive != 1 or not hasattr(qtpy.QtCore, 'PYQT_VERSION'):
            QApplication.exec_()
                          
   
        sys.exit ( "End of test")
    
    @time_it
    def show_inverted_proj(self):
        
        inverted_xy = np.sum(self.image_inv, 0)
        inverted_xz = np.sum(self.image_inv, 2)
        
        c_min = min( np.amin(np.amin(inverted_xy, 1), 0) , np.amin(np.amin(inverted_xz, 1), 0) )
        c_max = max(np.amax(np.amax(inverted_xy, 1), 0) , np.amax(np.amax(inverted_xz, 1), 0) )
        
        
        fig1, (ax1, ax2) =plt.subplots(2, 1, gridspec_kw={'height_ratios': [ 4, 1]})
        # fig1.clf()
        fig1.text(0.1,0.2, f'Inverted image projections (base: {self.base}, denoised: {self.denoised})')
        
        xy = ax1.imshow(inverted_xy.transpose(), cmap = 'gray', aspect = 1, vmin = c_min, vmax = c_max)
        ax1.set_xlabel('x (px)')
        ax1.set_ylabel('y (px)')
        cbar = fig1.colorbar(xy, ax = ax1, format='%.0e')
        cbar.ax.set_ylabel('Counts', rotation=270)
        
        xz = ax2.imshow(inverted_xz, cmap = 'gray', aspect = 12.82,  vmin = c_min, vmax = c_max) #aspect = 12.82
        ax2.set_xlabel('x (px)')
        ax2.set_ylabel('z (px)')
        # fig1.colorbar(xz, ax = ax1)
        
    @time_it    
    def show_inverted_xy(self):
        inverted_xy = np.sum(self.image_inv, 0)
        
        fig1=plt.figure()
        fig1.clf()
        fig1.suptitle(f'Inverted image XY projection (base: {self.base}, denoised: {self.denoised})')
        ax1=fig1.add_subplot(111)
        xy = ax1.imshow(inverted_xy.transpose(), cmap = 'gray', aspect = 1)
        ax1.set_xlabel('x (px)')
        ax1.set_ylabel('y (px)')
        cbar = fig1.colorbar(xy, ax = ax1, format='%.0e')
        cbar.ax.set_ylabel('Counts', rotation=270)
    
    @time_it
    def show_inverted_xz(self):
        inverted_xz = np.sum(self.image_inv, 2)
        
        dmdPx_to_sample_ratio = 1 # (um/px)
        aspect_xz = (self.ROI_s_z * dmdPx_to_sample_ratio / len(self.disp_freqs))/0.65
        
        # fig1=plt.figure( figsize = (3, 6) , constrained_layout=True) 
        fig1=plt.figure( constrained_layout=True) 
        fig1.clf()
        
        ax1=fig1.add_subplot(111)
        fig1.suptitle('Inverted image XZ projection')
        
        if not hasattr(self, 'denoise_w'):
            self.denoise_w = None
        
        ax1.set_title(f'Base: {self.base}, Denoised: {self.denoised} (w = {self.denoise_w}), clipped = {self.clipped}', fontsize = 10)
        xz = ax1.imshow(inverted_xz, cmap = 'gray', aspect = aspect_xz, interpolation = 'none') #aspect = 12.82 for 24 z pixels, aspect = 6.6558 for 61 z pixels, aspect = 11.80 for tests in 61px, aspect = 30 for testing in 24 px
        ax1.set_xlabel('x (px)')
        ax1.set_ylabel('z (px)')
        cbar = fig1.colorbar(xz, ax = ax1, shrink=0.5, format='%.0e')
        cbar.ax.set_ylabel('Counts', rotation=270)
    
        
    def show_inverted3D(self):
        
        # TODO: make work
        
        from pyqtgraph.Qt import QtCore, QtGui
        import pyqtgraph.opengl as gl
        
        # create qtgui
        app = QtGui.QApplication([])
        w = gl.GLViewWidget()
        w.orbit(256, 256)
        w.setCameraPosition(0, 0, 0)
        w.opts['distance'] = 200
        w.show()
        w.setWindowTitle('pyqtgraph example: GLVolumeItem')
        
        g = gl.GLGridItem()
        g.scale(20, 20, 1)
        w.addItem(g)
        
        
        v = gl.GLVolumeItem(self.image_inv, sliceDensity=1, smooth=False, glOptions='translucent')
        v.translate(-self.image_inv.shape[0]/2, -self.image_inv.shape[1]/2, -150)
        w.addItem(v)
        
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
            
    def save_inverted(self, outputFile):
        
        tiff.imsave(outputFile , np.uint16(self.image_inv.clip(min = 0)), append = False) 
        
    
 
if __name__ == "__main__" :
    

        #   ----  1 ----- 
        # file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/data_28_4_22/220428_124841_coherent_SVIM_phantom2_good.h5'
        #   ----  2 ----- 
        # file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/data_28_4_22/220428_124222_coherent_SVIM_phantom2_good.h5'
        #   ----  3 ----- 
        # file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/data_28_4_22/220428_125643_coherent_SVIM_phantom2_good.h5'
        #   ----  4 ----- 
        file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/data_28_4_22/220428_115143_coherent_SVIM_phantom2_good.h5'
        
        
        # file_name_h5 = file_name + '.h5'
        
        dataset = coherentSVIM_analysis(file_name)
        dataset.load_h5_file()
        
        
        dataset.merge_pos_neg()
        # dataset.setROI(814-40,  1132-40, 80)
        dataset.setROI(420,  524, 1000)
        # dataset.show_im_raw()
        
        dataset.choose_freq()
        
        base = 'cos'
        dataset.p_invert(base)
        dataset.show_inverted_xy()
        dataset.show_inverted_xz()
        
        #%% 
        
        dataset.denoise()
        # dataset.show_inverted_xy()
        dataset.show_inverted_xz()
        
        # %%
        # dataset.cut_negatives()
        dataset.show_inverted_xz()
        
        # base = 'sq'
        # dataset.p_invert(base)
        # dataset.show_inverted_xz()
        
        # base = 'sp_dct'
        # dataset.invert(base )
        # dataset.show_inverted_xz()
        
        
        
        
        
        
        
        # save_file = 'Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/data_28_4_22/220428_124841_coherent_SVIM_phantom2_good_inverted.tif'        
        
        # try:
        #     os.remove(save_file)
        # except:
        #     print('file name not found')
            
            
        # dataset.save_inverted(save_file)
        
        
        #%%
        # ===================================
        #              TV tests
        # ===================================
        
        
        
        
        line = dataset.image_inv[:,40,40]
        
        fig1, ax1 = plt.subplots(1,1)
        ax1.plot(line, '--', label = f'inverted pixel ({base})')
        ax1.set_xlabel('z (px)')
        
        # # ------------------------------------
        # # PYLOPS
        
        # import pylops
        # nz = len(line)
        # Iop = pylops.Identity(nz)
        
        # D2op = pylops.SecondDerivative(nz, edge=True)
        # lamda = 20
        
        # line_smooth_pylops_1 = pylops.optimization.leastsquares.RegularizedInversion(
        #     Iop, [D2op], line, epsRs=[np.sqrt(lamda / 2)], **dict(iter_lim=300)
        # )
        
        # ax1.plot(line_smooth_pylops_1, '-',label  = f'Smoothed pylops TV, lambda = {lamda}')
        # # ax1.legend()
        
        # lamda = 5
        
        # line_smooth_pylops_2 = pylops.optimization.leastsquares.RegularizedInversion(
        #     Iop, [D2op], line, epsRs=[np.sqrt(lamda / 2)], **dict(iter_lim=300)
        # )
        
        # ax1.plot(line_smooth_pylops_2,  '--', linewidth = 1, color = 'C1',label  = f'Smoothed pylops TV, lambda = {lamda}')
        
        # # ------------------------------------
        # # Sk-Image chambolle
        
        # from skimage.restoration import denoise_tv_chambolle
        # weight = max(line) * 0.1193 # cos
        weight = max(line) * 0.20439  #sq
        eps = 2e-4
        max_iter = 150
        
        t = time.time()
        
        reps= 1000
        for i in range(reps):
        
            line_smooth_skimage_1 = denoise_tv_chambolle(line, weight, eps, max_iter)
            
        print(f'time for single line: {(time.time() - t)/reps}s')
        
        ax1.plot(line_smooth_skimage_1, color = 'C2' , label  = f'Chambolle sk-image TV, weight = {weight}')
    
        weight = 300
        
        line_smooth_skimage_2 = denoise_tv_chambolle(line, weight)
        ax1.plot(line_smooth_skimage_2,'--', color = 'C2', linewidth = 1, label  = f'Chambolle sk-image TV, weight = {weight}')
        
        ax1.legend()
        
        
        
        # # ------------------------------------
        # # Sk-Image chambolle
        
        # from skimage.restoration import denoise_tv_bregman
        # weight = 100
        
        # line_smooth_skimage_1 = denoise_tv_bregman(line, weight)
        # ax1.plot(line_smooth_skimage_1, label  = f'Bregman sk-image TV, weight = {weight}')
        
        # # weight = 300
        
        # # line_smooth_skimage_2 = denoise_tv_bregman(line, weight)
        # # ax1.plot(line_smooth_skimage_2,'--', color = 'C3', linewidth = 1, label  = f'Bregman sk-image TV, weight = {weight}')
        
        # ax1.legend()
        
        
        
        
        
        
        
        