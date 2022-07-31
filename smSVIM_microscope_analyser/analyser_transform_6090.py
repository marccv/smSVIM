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
# import tifffile as tiff
import time
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 9})

# from skimage.restoration import denoise_tv_chambolle

import pylops
# from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import LinearOperator
# from scipy.optimize import least_squares
from numpy.linalg import lstsq

import sys
import os
import h5py
import pyqtgraph as pg
import qtpy.QtCore
from qtpy.QtWidgets import QApplication


from show_image import show_images_new_windows


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
    
    
    """
    
    Class for the analyis of a SVIM dataset.
    
    Methods:
        
        - load_h5_file:    loads raw images in self.imageRaw
        - invert:          inverts the raw images using the canonic inverse matrix of the problem,
                           some will have very high condition numbers -> use the methods p_invert or
                           lsqr_invert
        - p_invert:        inverts the raw images using the problem pseudo inverse with given condition number
        - lsqr_invert:     least square inversion
        - merge_pos_neg:   subtracts couples of raw complementary raw images
        - make_pos_neg:    
    
    """
    
    
    
    
    name = 'coherentSVIM_analysis'

    def __init__(self, fname, **kwargs):
        
        #default values
        self.params = {'base': 'hadam',
                       'pixel_size': 0.325, #(um/px)
                       'dmd_to_sample_ratio': 1.195, #(um/px)
                       'dark_counts': 100,
                       'PosNeg': True,
                       'select_ROI': False,
                       'denoise' : False,
                       'X0': 0,
                       'Y0': 0,
                       'delta_x' : 0,
                       'delta_y' : 0,
                       'mu': 0.01,
                       'lamda': 0.5,
                       'niter_out': 15,
                       'niter_in': 2,
                       'lsqr_niter': 5,
                       'lsqr_damp': 1e-4,
                       'single_volume_time_index' : 0,
                       'save_label': '',
                       'time_lapse_save_label': '',
                       'time_lapse_mode': 'sum',
                       'time_lapse_view': 0,
                       'time_lapse_plane': 0}
        
        # update any specified parameter
        for key, val in kwargs.items():
            self.params[key] = val
        
        self.file_path  = fname
        
        if not hasattr(self, 'plot_windows'):
            self.plot_windows = show_images_new_windows()
        
        
    @time_it   
    def load_h5_file(self, dataset_index = 0):
        
        self.imageRaw = get_h5_dataset(self.file_path, max(0,dataset_index)) 

    
    def show_im_raw(self):
                
        self.plot_windows.show_new_image(self.imageRaw.transpose(0,1,2), title="Raw image", ordinate = 'X', ascisse = 'Y', 
                   scale_ord = 0.65e-6, scale_asc = 0.65e-6)    
        
        if self.name == 'coherentSVIM_analysis':
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
    def make_pos_neg(self):
        
        # this method has problems when the contrast is not 1
        
        #TODO: this works only for walsh patterns
        self.imageRaw = 2*(self.imageRaw - np.mean(self.imageRaw[7:,:,:], 0))
    
    
    @time_it
    def setROI(self, **kwargs):
        
        # update any specified parameter
        for key, val in kwargs.items():
            self.params[key] = val
        
        self.imageRaw = self.imageRaw[:,
                                      self.params['X0'] : self.params['X0'] + self.params['delta_x'],
                                      self.params['Y0'] : self.params['Y0'] + self.params['delta_y']]
    
    @time_it
    def choose_freq(self, N = None):
        
        f_min = get_h5_attr(self.file_path, 'f_min')[0]
        f_max = get_h5_attr(self.file_path, 'f_max')[0]
        ROI_s_z = get_h5_attr(self.file_path, 'ROI_s_z')[0]
        self.ROI_s_z = ROI_s_z
        
        freqs = np.linspace(f_min, f_max, int(2*(f_max - f_min) + 1),dtype = float)
                
        if f_min == 0.0:
            disp_freqs = [0]
            for freq in freqs[1:]:
                period = int(ROI_s_z/freq)
                disp_freqs.append(ROI_s_z/period)
        else:
            disp_freqs = []
            for freq in freqs:
                period = int(ROI_s_z/freq)
                disp_freqs.append(ROI_s_z/period)
                
        
        self.imageRaw = self.imageRaw[0:N, :, :]
        self.disp_freqs = disp_freqs[0:N]
        
        # to eliminate copies of the same frequency
        mask = np.append(np.diff(disp_freqs)!= 0, True)
        
        # self.imageRaw = self.imageRaw[mask, :, :]
        print(f'image Raw shape: {self.imageRaw.shape}')
        self.disp_freqs = np.array(disp_freqs)[mask]
        
        
        
    
    @time_it    
    def invert(self, **kwargs):
        
        # update any specified parameter
        for key, val in kwargs.items():
            self.params[key] = val
        
        
        
        if self.params['base'] == 'cos':
            self.transform = t_6090.dct_6090(self.disp_freqs)
            self.transform.create_space()
            self.transform.create_matrix_cos()
            self.transform.compute_inverse()
            self.image_inv = np.tensordot(self.transform.inv_matrix ,  self.imageRaw , axes=([1],[0]))
            
        elif self.params['base'] == 'sq':
            self.transform = t_6090.dct_6090(self.disp_freqs)
            self.transform.create_space()
            self.transform.create_matrix_sq()
            self.transform.compute_inverse()
            self.image_inv = np.tensordot(self.transform.inv_matrix ,  self.imageRaw , axes=([1],[0]))
            
        elif self.params['base'] == 'hadam':
            # normal hadamard matrix is orthogonal
            
            self.ROI_s_z = get_h5_attr(self.file_path, 'ROI_s_z')[0]
            self.params['had_pat_num'] = int(get_h5_attr(self.file_path, 'had_pat_num')[0])
            
            inv_matrix = (1/self.params['had_pat_num'])*t_6090.create_hadamard_matrix(self.params['had_pat_num'], 'hadam')
            self.image_inv = np.tensordot(inv_matrix ,  self.imageRaw , axes=([1],[0]))
            
        elif self.params['base'] == 'sp_dct':
            dct_coeff = self.imageRaw
            dct_coeff[0,:,:] *= 1/np.sqrt(2)  # I rescale the cw illumination >> It just shifts the inverted image towards more negative values
            self.image_inv = sp_fft.idct(dct_coeff, type = 2, axis = 0, norm = 'ortho')
        
        
        
        self.denoised = False
        self.clipped = False
        
        
    
    @time_it        
    def p_invert(self,  **kwargs):
        
        '''
        Inverts the raw image using the the matrix pseudoinverse with rcond = 10
        '''
        # update any specified parameter
        for key, val in kwargs.items():
            self.params[key] = val
        
        self.transform = t_6090.dct_6090(self.disp_freqs)
        self.transform.create_space()
        
        if self.params['base'] == 'cos':
            self.transform.create_matrix_cos()
            self.transform.compute_pinv()
            self.image_inv = np.tensordot(self.transform.pinv_matrix ,  self.imageRaw , axes=([1],[0]))
            
        elif self.params['base'] == 'sq':
            self.transform.create_matrix_sq()
            self.transform.compute_pinv()
            self.image_inv = np.tensordot(self.transform.pinv_matrix ,  self.imageRaw , axes=([1],[0]))
        
        self.denoised = False
        self.clipped = False
    
    
    
    @time_it
    def lsqr_invert(self, **kwargs):
        
        '''
        Inverts the raw image using least squares methods
        '''
        # update any specified parameter
        for key, val in kwargs.items():
            self.params[key] = val
    
        
        
        if self.params['base'] == 'cos':
            self.choose_freq()
            self.transform = t_6090.dct_6090(self.disp_freqs)
            self.transform.create_space()
            self.transform.create_matrix_cos()
            matrix = self.transform.matrix
            
        elif self.params['base'] == 'sq':
            self.choose_freq()
            self.transform = t_6090.dct_6090(self.disp_freqs)
            self.transform.create_space()
            self.transform.create_matrix_sq()
            matrix = self.transform.matrix
            
        else:
            # hadamard type base
            self.ROI_s_z = get_h5_attr(self.file_path, 'ROI_s_z')[0]
            self.params['had_pat_num'] = int(get_h5_attr(self.file_path, 'had_pat_num')[0])
            
            matrix = t_6090.create_hadamard_matrix(self.params['had_pat_num'], self.params['base'])
            
        
        
        if not self.params['PosNeg'] and not self.params['make_posneg']:
            matrix[matrix <0] = 0
            
            # subtract dark counts
            self.imageRaw -= self.params['dark_counts']
    
    
    
    
        matrix = matrix.astype(float)
        Nz = matrix.shape[1]
        nz,ny,nx = self.imageRaw.shape
        
        self.image_inv,_,_,_ = lstsq(matrix, self.imageRaw.reshape( nz, int(ny*nx)), rcond = None)
        
        # print(type(self.image_inv))
        
        self.image_inv = self.image_inv.reshape(Nz,ny,nx)
        
        
        
    
    # @time_it
    # def denoise(self, weight = 1000):
        
    #     shape = self.image_inv.shape
    #     self.param = 'Chambolle 1D denoise on already inverted image with linear weigth'
        
    #     if self.params['base'] == 'cos':
    #         ratio = 0.1193
    #     elif self.params['base'] == 'sq':
    #         ratio = 0.20439
        
    #     for i in range(shape[1]):
    #         for j in range(shape[2]):
                
    #             weight = max(self.image_inv[:,i,j]) * ratio
    #             self.image_inv[:,i,j] = denoise_tv_chambolle(self.image_inv[:,i,j], weight)
        
    #     self.denoised = True
    
    
    @time_it
    def invert_and_denoise1D_no_for(self, **kwargs):
        
        # update any specified parameter
        for key, val in kwargs.items():
            self.params[key] = val
        
        if self.params['base'] == 'cos':
            self.transform = t_6090.dct_6090(self.disp_freqs)
            self.transform.create_space()
            self.transform.create_matrix_cos()
            M = self.transform.matrix
            
        elif self.params['base'] == 'sq':
            self.transform = t_6090.dct_6090(self.disp_freqs)
            self.transform.create_space()
            self.transform.create_matrix_sq()
            M = self.transform.matrix
            
        elif self.params['base'] == 'hadam':
            self.ROI_s_z = get_h5_attr(self.file_path, 'ROI_s_z')[0]
            self.params['had_pat_num'] = get_h5_attr(self.file_path, 'had_pat_num')[0]
            M = t_6090.create_hadamard_matrix(self.params['had_pat_num'], 'hadam')
        
        nz,ny,nx = self.imageRaw.shape
        shape = (nz,ny,nx)
        M = M.astype(float)
        Nz = M.shape[1]
        
        
        def Op(v):
            v = v.reshape( Nz, int(len(v)/Nz))
            return (M@v).ravel()
        
        def Op_t(v):
            v = v.reshape( nz, int(len(v)/nz))
            return (M.transpose()@v).ravel()
        
        Op_s = LinearOperator((nz*nx*ny, Nz*nx*ny), matvec = Op, rmatvec  = Op_t, dtype = float)
        Op_s = pylops.LinearOperator(Op_s)
        
        Dop = pylops.FirstDerivative(Nz*ny*nx, (Nz, ny, nx),  0, edge=True, kind="backward")
        
        self.params['denoise_type'] =  '1D'
        print(shape)
        
        t = time.time()
        
        self.image_inv, _ = pylops.optimization.sparsity.SplitBregman(
                                    Op_s,
                                    [Dop],
                                    self.imageRaw.ravel(),
                                    self.params['niter_out'],
                                    self.params['niter_in'],
                                    mu = self.params['mu'],
                                    epsRL1s=[self.params['lamda']],
                                    tol=1e-4,
                                    tau=1.0,
                                    **dict(iter_lim=self.params['lsqr_niter'], damp=self.params['lsqr_damp'])
                                )
        print(f'time for one line: {(time.time()  - t)/(shape[1] * shape[2])}')
        
        
        self.image_inv = self.image_inv.reshape(Nz,ny,nx)
        self.image_inv = self.image_inv.transpose(0,2,1)
        
        self.clipped = False
        
    
    @time_it
    def invert_and_denoise3D_v2(self, **kwargs):
        
        # update any specified parameter
        for key, val in kwargs.items():
            self.params[key] = val
        
        
    
        if self.params['base'] == 'cos':
            self.transform = t_6090.dct_6090(self.disp_freqs)
            self.transform.create_space()
            self.transform.create_matrix_cos()
            M = self.transform.matrix
            
        elif self.params['base'] == 'sq':
            self.transform = t_6090.dct_6090(self.disp_freqs)
            self.transform.create_space()
            self.transform.create_matrix_sq()
            M = self.transform.matrix
            
        elif self.params['base'] == 'hadam':
            self.ROI_s_z = get_h5_attr(self.file_path, 'ROI_s_z')[0]
            self.params['had_pat_num'] = get_h5_attr(self.file_path, 'had_pat_num')[0]
            M = t_6090.create_hadamard_matrix(self.params['had_pat_num'], 'hadam')
        
        nz,ny,nx = self.imageRaw.shape
        shape = (nz,ny,nx)
        M = M.astype(float)
        
        if not self.params['PosNeg'] and not self.params['make_posneg']:
            M[M <0] = 0
            
            # subtract dark counts
            self.imageRaw -= self.params['dark_counts']
        
        Nz = M.shape[1]
        
        def Op(v):
            v = v.reshape( Nz, int(len(v)/Nz))
            return (M@v).ravel()
        
        def Op_t(v):
            v = v.reshape( nz, int(len(v)/nz))
            return (M.transpose()@v).ravel()
        
        Op_s = LinearOperator((nz*nx*ny, Nz*nx*ny), matvec = Op, rmatvec  = Op_t,dtype = float)
        Op_s = pylops.LinearOperator(Op_s)
        
        
        
        Dop = [
            pylops.FirstDerivative(Nz*ny*nx, (Nz, ny, nx),  0, edge=True, kind="backward"),
            pylops.FirstDerivative(Nz*ny*nx, (Nz, ny, nx),  1, edge=True, kind="backward"),
            pylops.FirstDerivative(Nz*ny*nx, (Nz, ny, nx),  2, edge=True, kind="backward")
        ]
        
        print(shape)
        
        t = time.time()
        
        self.image_inv, _ = pylops.optimization.sparsity.SplitBregman(
                                    Op_s,
                                    Dop,
                                    self.imageRaw.ravel(),
                                    self.params['niter_out'],
                                    self.params['niter_in'],
                                    mu = self.params['mu'],
                                    epsRL1s = [self.params['lamda']]*3,
                                    tol=1e-4,
                                    tau=1.0,
                                    **dict(iter_lim=self.params['lsqr_niter'], damp=self.params['lsqr_damp'])
                                )
        print(f'time for one line: {(time.time()  - t)/(shape[1] * shape[2])}')
        print(Nz,ny, nx)
        
        self.image_inv = self.image_inv.reshape(Nz,ny,nx)
        # self.image_inv = self.image_inv.transpose(0,2,1)
        
        self.params['denoise_type'] =  '3D'
        self.clipped = False
    

    
    
    
    
    
    
# =============================================================================
#     Visualize volumes
# =============================================================================
    
    
    
    
    
    @time_it
    def cut_negatives(self):
        
        self.image_inv = self.image_inv.clip(min = 0)
        self.clipped = True
    
    def show_inverted(self):
        
        # TODO make  the aspect ratio of the displayedimage match the sampled volume aspect ratio
        
        pg.image(self.image_inv, title= f"Inverted volume (base: {self.params['base']})")        
        
        if self.name == 'coherentSVIM_analysis':
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
        fig1.text(0.1,0.2, f'Inverted volume projections\n{self.params}')
        
        xy = ax1.imshow(inverted_xy.transpose(), cmap = 'gray', aspect = 1, vmin = c_min, vmax = c_max)
        ax1.set_xlabel('x (px)')
        ax1.set_ylabel('y (px)')
        cbar = fig1.colorbar(xy, ax = ax1, format='%.0e')
        cbar.ax.set_ylabel('Counts', rotation=270)
        
        # xz = ax2.imshow(inverted_xz, cmap = 'gray', aspect = 12.82,  vmin = c_min, vmax = c_max) #aspect = 12.82
        ax2.set_xlabel('x (px)')
        ax2.set_ylabel('z (px)')
        # fig1.colorbar(xz, ax = ax1)
        
    @time_it    
    def show_inverted_xy(self, plane = 'sum'):
        
        if plane == 'sum':
            inverted_xy = np.sum(self.image_inv, 0)
        else:
            inverted_xy = self.image_inv[plane,:,:] # to show just one xy plane
        
        fig1=plt.figure()
        fig1.clf()
        fig1.suptitle(f'Inverted volume XY projection\n{self.params}')
        ax1=fig1.add_subplot(111)
        xy = ax1.imshow(inverted_xy.transpose(), cmap = 'gray', aspect = 1)
        ax1.set_xlabel('x (px)')
        ax1.set_ylabel('y (px)')
        cbar = fig1.colorbar(xy, ax = ax1, format='%.0e')
        cbar.ax.set_ylabel('Counts', rotation=270)
    
    @time_it
    def show_inverted_xz(self, plane = 'sum', **kwargs):
        
        # if kwargs is not None:
        #     for key, value in kwargs.items():
        #         print( key, '==>', value)
        
        if plane == 'sum':
            inverted_xz = np.sum(self.image_inv, 2)
        else:
            inverted_xz = self.image_inv[:,:,plane] # to show just one xz plane
        
        aspect_xz = (self.ROI_s_z * self.params['dmd_to_sample_ratio'] / self.image_inv.shape[0] )/self.params['pixel_size']
        
        # aspect_xz = 0.5
        
        # fig1=plt.figure( figsize = (3, 6) , constrained_layout=True) 
        if __name__ == 'coherentSVIM_analysis':
        
            fig1=plt.figure( constrained_layout=False) 
            fig1.clf()
            ax1=fig1.add_subplot(111)
            
        else:
            
            fig1 = kwargs.get('fig')
            ax1 = kwargs.get('ax')
            
            # print(fig1 is None)
            # print(ax1 is None)
        
        
        fig1.suptitle(f'Inverted volume XZ projection\n{self.params}')
        
        xz = ax1.imshow(inverted_xz, cmap = 'gray', aspect = aspect_xz, interpolation = 'none') #aspect = 12.82 for 24 z pixels, aspect = 6.6558 for 61 z pixels, aspect = 11.80 for tests in 61px, aspect = 30 for testing in 24 px
        ax1.set_xlabel('x (px)')
        ax1.set_ylabel('z (px)')
        cbar = fig1.colorbar(xz, ax = ax1, shrink=1, format='%.0e')
        cbar.ax.set_ylabel('Counts', rotation=270)
    

    @time_it
    def save_inverted(self):
        
        try:
            head, tail = os.path.split(self.file_path)
            
            newpath = self.file_path[:-3] + '_ANALYSED'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            
            if len(self.params["save_label"]) >0:
                fname = os.path.join(newpath, f'volume_{self.params["single_volume_time_index"]}_inverted_{self.params["save_label"]}.h5')
            else:
                fname = os.path.join(newpath, f'volume_{self.params["single_volume_time_index"]}_inverted.h5')
            
            while os.path.exists(fname):
                fname = fname[:-3] + '_bis.h5'
            
            parent = h5py.File(fname,'w')
    
            # create groups
            analysis_parameters = parent.create_group('analysis_parameters') 
            
            for key, val in self.params.items():
                analysis_parameters.attrs[key] = val
     
            # create a dataset
            name = 't000/c000/' + tail[:-3]
            h5dataset = parent.create_dataset(name = name, shape=self.image_inv.shape, data = self.image_inv)
            h5dataset.dims[0].label = "z"
            h5dataset.dims[1].label = "y"
            h5dataset.dims[2].label = "x"
            
            self.ROI_s_z = get_h5_attr(self.file_path, 'ROI_s_z')[0]
            z_sample_period = self.ROI_s_z * self.params['dmd_to_sample_ratio'] / self.image_inv.shape[0] 
            h5dataset.attrs['element_size_um'] =  [z_sample_period, self.params['pixel_size'], self.params['pixel_size']]


        finally:
            parent.close()
            
            
            
            
# =============================================================================
#       Time lapse
# =============================================================================
            
            
    def invert_and_save_complete_tl(self, **kwargs):
        for key, val in kwargs.items():
            self.params[key] = val
        
        print(self.params['time_lapse_mode'])
        
        
        try:
            try:
                self.params['time_frames_n'] = get_h5_attr(self.file_path, 'real_time_frames_n')[0]
            except:
                self.params['time_frames_n'] = get_h5_attr(self.file_path, 'time_frames_n')[0]
        except:
            print('>> Warning: Could not find the number of time frames.')
            
        else: # if there are no errors
        
        
            # open the H5 file
        
            try:
                head, tail = os.path.split(self.file_path)
                
                newpath = self.file_path[:-3] + '_ANALYSED'
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                
                
                fname = os.path.join(newpath, 'time_lapse_inverted_complete.h5')
                
                while os.path.exists(fname):
                    fname = fname[:-3] + '_bis.h5'
                
                parent = h5py.File(fname,'w')
        
                # create groups
                analysis_parameters = parent.create_group('analysis_parameters') 
                
                for key, val in self.params.items():
                    analysis_parameters.attrs[key] = val
         
                
                
            
                
                for time_index in range(self.params['time_frames_n']):
                    
                    self.load_h5_file(time_index)
                    
                    if self.select_ROI: self.setROI()
                    if self.params['PosNeg'] : self.merge_pos_neg()
                    if self.params['make_posneg']: self.make_pos_neg()
            
                    if not self.denoise:
                        # try:
                            
                        if self.params['base'] != 'hadam' or not self.params['PosNeg']:
                            self.lsqr_invert()
                        else:
                            self.invert()
                                
                    else:
                        if self.params['base'] == 'cos' or self.params['base'] == 'sq':
                            self.choose_freq()
                            
                        self.invert_and_denoise3D_v2()   
                        
                    # create a dataset
                    name = f't{time_index:04d}/c0000/image'
                    h5dataset = parent.create_dataset(name = name, shape=self.image_inv.shape, data = self.image_inv)
                    h5dataset.dims[0].label = "z"
                    h5dataset.dims[1].label = "y"
                    h5dataset.dims[2].label = "x"
                    
                    
                    
                    depth_z = (self.ROI_s_z * self.params['dmd_to_sample_ratio']/ self.image_inv.shape[0] )
                    h5dataset.attrs['element_size_um'] =  [depth_z, self.params['pixel_size'], self.params['pixel_size']]
                
            finally:
                parent.close()
            
                
       
            
            
            
    def invert_time_lapse(self, **kwargs):
        
        # update any specified parameter
        for key, val in kwargs.items():
            self.params[key] = val
        
        print(self.params['time_lapse_mode'])
        
        
        if kwargs.get('progress_bar') is not None:
            progress_bar = kwargs.get('progress_bar')
            progress_bar.setValue(0)
        
        try:
            try:
                self.params['time_frames_n'] = get_h5_attr(self.file_path, 'real_time_frames_n')[0]
            except:
                self.params['time_frames_n'] = get_h5_attr(self.file_path, 'time_frames_n')[0]
        except:
            print('>> Warning: Could not find the number of time frames.')
            
        else: # if there are no errors
                
            self.tl_stack = []
            
            for time_index in range(self.params['time_frames_n']):
                
                self.load_h5_file(time_index)
                
                if self.select_ROI: self.setROI()
                if self.params['PosNeg']: self.merge_pos_neg()
                if self.params['make_posneg']: self.make_pos_neg()
        
                if not self.params['denoise']:
                    # try:
                        
                    if self.params['base'] == 'hadam' and self.params['PosNeg']:
                        self.invert()
                    else:
                        self.lsqr_invert()
                            
                else:
                    if self.params['base'] == 'cos' or self.params['base'] == 'sq':
                        self.choose_freq()
                        
                    self.invert_and_denoise3D_v2() 
                
                        
            
                
                if self.params['time_lapse_mode'] == 'max':
                    
                    if self.params['time_lapse_view'] == 0:   #xy
                        self.tl_stack.append(np.max(self.image_inv, 0))  
                        
                    elif self.params['time_lapse_view'] == 1: #xz
                        self.tl_stack.append(np.max(self.image_inv, 2))  
                        
                    elif self.params['time_lapse_view'] == 2: #yz
                        self.tl_stack.append(np.max(self.image_inv, 1))   
                    
                    
                elif self.params['time_lapse_mode'] == 'ave':
                    if self.params['time_lapse_view'] == 0:   #xy
                        self.tl_stack.append(np.mean(self.image_inv, 0))  
                        
                    elif self.params['time_lapse_view'] == 1: #xz
                        self.tl_stack.append(np.mean(self.image_inv, 2))  
                        
                    elif self.params['time_lapse_view'] == 2: #yz
                        self.tl_stack.append(np.mean(self.image_inv, 1)) 
                    
                elif self.params['time_lapse_mode'] == 'plane':
                
                
                    if self.params['time_lapse_view'] == 0:
                        self.tl_stack.append(self.image_inv[self.params['time_lapse_plane'],:,:])
                    elif self.params['time_lapse_view'] == 1:
                        self.tl_stack.append(self.image_inv[:,:,self.params['time_lapse_plane']])
                    elif self.params['time_lapse_view'] == 2:
                        self.tl_stack.append(self.image_inv[:,self.params['time_lapse_plane'],:])
            
            self.tl_stack = np.array(self.tl_stack)
       
            
        
    def show_time_lapse(self):
        
        
        depth_z = (self.ROI_s_z * self.params['dmd_to_sample_ratio']/ self.image_inv.shape[0] ) *1e-6 #(m/px)
        width_xy = self.params['pixel_size']*1e-6  #(m/px)
        
        if self.params['time_lapse_view'] == 0:   #xy
            title= f"Inverted Time Lapse XY {self.params['time_lapse_mode']} (base: {self.params['base']})"
            self.plot_windows.show_new_image(self.tl_stack, title= title, ordinate = 'X', ascisse = 'Y', 
                       scale_ord = width_xy, scale_asc = width_xy)  
            
        elif self.params['time_lapse_view'] == 1: #xz
            title= f"Inverted Time Lapse XZ {self.params['time_lapse_mode']} (base: {self.params['base']})"
            self.plot_windows.show_new_image(self.tl_stack.transpose((0,2,1)), title= title, ordinate = 'X', ascisse = 'Z', 
                       scale_ord = width_xy, scale_asc = depth_z )  
            
        elif self.params['time_lapse_view'] == 2: #yz
            title= f"Inverted Time Lapse YZ {self.params['time_lapse_mode']} (base: {self.params['base']})"
            self.plot_windows.show_new_image(self.tl_stack, title= title, ordinate = 'Z', ascisse = 'Y', 
                       scale_ord = depth_z, scale_asc = width_xy)  
            
        
        
        # pg.image(self.tl_stack, title= f"Inverted Time Lapse (base: {self.params['base']})")        
        
        if self.name == 'coherentSVIM_analysis':
            #keeps the window open running a QT application
            if sys.flags.interactive != 1 or not hasattr(qtpy.QtCore, 'PYQT_VERSION'):
                QApplication.exec_()
            sys.exit ( "End of test")
    
    def save_time_lapse(self):
        
        try:
            head, tail = os.path.split(self.file_path)
            
            newpath = self.file_path[:-3] + '_ANALYSED'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
                
            view = ['XY', 'XZ', 'YZ'][self.params['time_lapse_view']]
            
            if len(self.params["time_lapse_save_label"]) >0:
                fname = os.path.join(newpath, f'time_lapse_inverted_{self.params["time_lapse_mode"]}_{view}_{self.params["time_lapse_save_label"]}.h5')
            else:
                fname = os.path.join(newpath, f'time_lapse_inverted_{self.params["time_lapse_mode"]}_{view}.h5')
            
            while os.path.exists(fname):
                fname = fname[:-3] + '_bis.h5'
            
            parent = h5py.File(fname,'w')
    
            # create groups
            analysis_parameters = parent.create_group('analysis_parameters') 
            
            for key, val in self.params.items():
                analysis_parameters.attrs[key] = val
     
            # create a dataset
            name = 't000/c000/' + tail[:-3]
            h5dataset = parent.create_dataset(name = name, shape=self.tl_stack.shape, data = self.tl_stack)
            h5dataset.dims[0].label = "t"
            h5dataset.dims[1].label = "y"
            h5dataset.dims[2].label = "x"
            
            h5dataset.attrs['element_size_um'] =  [1, self.params['pixel_size'], self.params['pixel_size']]
            
            
            depth_z = (self.ROI_s_z * self.params['dmd_to_sample_ratio']/ self.image_inv.shape[0] )
            
            
            
            if self.params['time_lapse_view'] == 0:   #xy
                h5dataset.dims[0].label = "t"
                h5dataset.dims[1].label = "y"
                h5dataset.dims[2].label = "x"
                
                h5dataset.attrs['element_size_um'] =  [1, self.params['pixel_size'], self.params['pixel_size']]
                
            elif self.params['time_lapse_view'] == 1: #xz
                h5dataset.dims[0].label = "t"
                h5dataset.dims[1].label = "z"
                h5dataset.dims[2].label = "x"
                
                h5dataset.attrs['element_size_um'] =  [1, depth_z, self.params['pixel_size']]
                
            elif self.params['time_lapse_view'] == 2: #yz
                h5dataset.dims[0].label = "t"
                h5dataset.dims[1].label = "y"
                h5dataset.dims[2].label = "z"
                
                h5dataset.attrs['element_size_um'] =  [1, self.params['pixel_size'], depth_z]
            
        finally:
            parent.close()

        
        
#%%    


# The use of this script has been made easier with the GUI found in "analyser_6090_app.py" (https://github.com/marccv/coherentSVIM)

 
if __name__ == "__main__" :
    

        file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/220523_cuma_fluo_test/220523_113202_coherent_SVIM_no_diff_300ul_transp.h5'
        
        
        dataset = coherentSVIM_analysis(file_name)
        dataset.load_h5_file()
        
        
        dataset.merge_pos_neg()
        # num_frames = dataset.imageRaw.shape[0]
        # dataset.image_inv =  dataset.imageRaw[np.linspace(0, num_frames -2, int(num_frames/2), dtype = 'int'), :, :].copy()

        # dataset.setROI(200, 0 , 600, 1024)

        # dataset.show_im_raw()
        
        # dataset.choose_freq() # also removes any duplicate in frequency
        
        #%% invert the raw image
        
        base = 'cos'
        mu = 0.01
        lamda = 0.5
        niter_out = 15
        niter_in = 2
        lsqr_niter = 5
        lsqr_damp = 1e-4
        
        # dataset.p_invert(base = base)
        
        # dataset.invert_and_denoise1D_no_for(base = base, lamda = lamda, niter_out = niter_out,
                                # niter_in = niter_in, lsqr_niter = lsqr_niter, lsqr_damp = lsqr_damp)
        # invert_and_denoise3D_v2(base = base, lamda = lamda, niter_out = niter_out,
                                # niter_in = niter_in, lsqr_niter = lsqr_niter, lsqr_damp = lsqr_damp)
        
        dataset.show_inverted()
        
        # dataset.show_inverted_xy()
        # dataset.show_inverted_xz()
        
        #%% show a single line profile along the z direction
        

        fig, ax = plt.subplots()
        ax.plot(dataset.image_inv[:,20,20],'x-', label = f'{dataset.params}')
        ax.legend()
        
        
        
        #%% save the inverted image
        dataset.image_inv = dataset.imageRaw
        dataset.params['save_label'] = 'pos_neg_merged'
        dataset.params['single_volume_time_index'] = 0
        dataset.ROI_s_z = 600
        dataset.save_inverted()
            
            
        #%% load a saved reconstructed image
        
        # import h5py
        
        # import sys
        # import pyqtgraph as pg
        # import qtpy.QtCore
        # from qtpy.QtWidgets import QApplication
        
        # fname = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/220509_163108_coherent_SVIM_plant1_elongation_diffuser_INVERTED_denoise_05.h5'
        # fname = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/220509_163108_coherent_SVIM_plant1_elongation_diffuser_INVERTED_2.h5'
        
        
        
        # h5file = h5py.File(fname,'r')
        
        ricostruita = coherentSVIM_analysis(fname)
        ricostruita.load_h5_file()
        # ricostruita.setROI(400, 0 , 200, 1024)
        
        im = ricostruita.imageRaw.copy()
        
        # xy planes
        # pg.image(im, title="Riconstructed image")        
        
        
        # xz planes
        pg.image(im.transpose(2,1,0), title="Riconstructed image",  *dict(transform = 0.1)  )      
               
        #keeps the window open running a QT application
        if sys.flags.interactive != 1 or not hasattr(qtpy.QtCore, 'PYQT_VERSION'):
            QApplication.exec_()
                          
   
        sys.exit ( "End of test")
        
        
        #%% invert a series of time frames and save for each of them a sigle image (given xy plane or sum along z) in the list called stack

        
        stack = []
        
        for i in range(1,97):
            
            fname_temp = str(fname + str(i) + ').h5')
            # print(fname_temp)
            
            
            dataset = coherentSVIM_analysis(fname_temp)
            dataset.load_h5_file()
            dataset.merge_pos_neg()
            dataset.choose_freq() # also removes any duplicate in frequency
            
            base = 'cos'            
            dataset.p_invert(base)
            
            # stack.append(dataset.image_inv[12,:,:])
            stack.append(np.sum(dataset.image_inv,0))
        
        
        # import h5py
        
        #%% save in a H5 file the stack we have just created
        
        stack = np.array(stack)
        
        try:
        
            fname = "D:\\LabPrograms\\ScopeFoundry_POLIMI\\smSVIM_microscope_analyser\\analysed\\220511\\ro_gfp\\time_laps_rog_leaf.h5"
            parent = h5py.File(fname,'w')
    
            # create groups
            results = parent.create_group('level1') 
            
    
            # create a dataset
    
            parent.create_dataset('inverted_time_laps_after_stimulus', shape=stack.shape, data = stack)
            # parent['voltage2'] = signalA
    
            # create attributes (not in the parents)
            results.attrs['sample'] = 'GFP'
            results.attrs['microscope_type'] = 'SIM'
            results.attrs['denoise_weight'] = 0

        finally:
            parent.close()
            
        