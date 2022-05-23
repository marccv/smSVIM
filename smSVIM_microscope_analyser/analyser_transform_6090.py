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
plt.rcParams.update({'font.size': 9})

# from skimage.restoration import denoise_tv_chambolle

import pylops
# from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import LinearOperator

import sys
import os
import h5py
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

    def __init__(self, fname, params = None ):
        
        if params is None:
            self.params = {'base': 'cos',
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
                           'lsqr_damp': 1e-4}
        else:
            self.params = params
        
        self.file_path  = fname
        
        
    @time_it   
    def load_h5_file(self, dataset_index = 0):
        
        self.imageRaw = get_h5_dataset(self.file_path, max(0,dataset_index)) 

    
    def show_im_raw(self):
        
        pg.image(self.imageRaw, title="Raw image")    
        
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
        
        self.transform = t_6090.dct_6090(self.disp_freqs)
        self.transform.create_space()
        
        if self.params['base'] == 'cos':
            self.transform.create_matrix_cos()
            self.transform.compute_inverse()
            self.image_inv = np.tensordot(self.transform.inv_matrix ,  self.imageRaw , axes=([1],[0]))
            
        elif self.params['base'] == 'sq':
            self.transform.create_matrix_sq()
            self.transform.compute_inverse()
            self.image_inv = np.tensordot(self.transform.inv_matrix ,  self.imageRaw , axes=([1],[0]))
            
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
    def denoise(self, weight = 1000):
        
        shape = self.image_inv.shape
        self.param = 'Chambolle 1D denoise on already inverted image with linear weigth'
        
        if self.params['base'] == 'cos':
            ratio = 0.1193
        elif self.params['base'] == 'sq':
            ratio = 0.20439
        
        for i in range(shape[1]):
            for j in range(shape[2]):
                
                weight = max(self.image_inv[:,i,j]) * ratio
                # self.image_inv[:,i,j] = denoise_tv_chambolle(self.image_inv[:,i,j], weight)
        
        self.denoised = True
    
    
    @time_it
    def invert_and_denoise1D_no_for(self, **kwargs):
        
        # update any specified parameter
        for key, val in kwargs.items():
            self.params[key] = val
        
        self.transform = t_6090.dct_6090(self.disp_freqs)
        self.transform.create_space()
    
        if self.params['base'] == 'cos':
            self.transform.create_matrix_cos()
            
        elif self.params['base'] == 'sq':
            self.transform.create_matrix_sq()
        
        nz,ny,nx = self.imageRaw.shape
        shape = (nz,ny,nx)
        M = self.transform.matrix
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
        
        self.params['denoise_type': '1D']
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
        
        
        self.image_inv = dataset.image_inv.reshape(Nz,ny,nx)
        self.image_inv = self.image_inv.transpose(0,2,1)
        
        self.clipped = False
        
    
    @time_it
    def invert_and_denoise3D_v2(self, **kwargs):
        
        # update any specified parameter
        for key, val in kwargs.items():
            self.params[key] = val
        
        self.transform = t_6090.dct_6090(self.disp_freqs)
        self.transform.create_space()
    
        if self.params['base'] == 'cos':
            self.transform.create_matrix_cos()
            
        elif self.params['base'] == 'sq':
            self.transform.create_matrix_sq()
        
        nz,ny,nx = self.imageRaw.shape
        shape = (nz,ny,nx)
        M = self.transform.matrix
        M = M.astype(float)
        
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
        
        
        self.image_inv = self.image_inv.reshape(Nz,ny,nx)
        # self.image_inv = self.image_inv.transpose(0,2,1)
        
        self.params['denoise_type': '1D']
        self.clipped = False
    

    
    
    
    @time_it
    def cut_negatives(self):
        
        self.image_inv = self.image_inv.clip(min = 0)
        self.clipped = True
    
    def show_inverted(self):
        
        # TODO make  the aspect ratio of the displayedimage match the sampled volume aspect ratio
        
        pg.image(self.image_inv, title= f"Inverted image (base: {self.params['base']})")        
        
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
        fig1.text(0.1,0.2, f'Inverted image projections\n{self.params}')
        
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
    def show_inverted_xy(self, plane = 'sum'):
        
        if plane == 'sum':
            inverted_xy = np.sum(self.image_inv, 0)
        else:
            inverted_xy = self.image_inv[plane,:,:] # to show just one xy plane
        
        fig1=plt.figure()
        fig1.clf()
        fig1.suptitle(f'Inverted image XY projection\n{self.params}')
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
        
        dmdPx_to_sample_ratio = 1.247 # (um/px)
        aspect_xz = (self.ROI_s_z * dmdPx_to_sample_ratio / self.image_inv.shape[0] )/0.65
        
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
        
        
        fig1.suptitle(f'Inverted image XZ projection\n{self.params}')
        
        xz = ax1.imshow(inverted_xz, cmap = 'gray', aspect = aspect_xz, interpolation = 'none') #aspect = 12.82 for 24 z pixels, aspect = 6.6558 for 61 z pixels, aspect = 11.80 for tests in 61px, aspect = 30 for testing in 24 px
        ax1.set_xlabel('x (px)')
        ax1.set_ylabel('z (px)')
        cbar = fig1.colorbar(xz, ax = ax1, shrink=1, format='%.0e')
        cbar.ax.set_ylabel('Counts', rotation=270)
    

    # @time_it
    def save_inverted(self):
        
        try:
            head, tail = os.path.split(self.file_path)
            
            newpath = self.file_path[:-3] + '_ANALYSED'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            
            index_key = 'single_volume_time_index'
            label_key = 'save_label'
            
            if len(self.params[label_key]) >0:
                fname = os.path.join(newpath, f'volume_{self.params[index_key]}_inverted_{self.params[label_key]}.h5')
            else:
                fname = os.path.join(newpath, f'volume_{self.params[index_key]}_inverted.h5')
            
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
            
            dmdPx_to_sample_ratio = 1.247 # (um/px)
            z_sample_period = self.ROI_s_z * dmdPx_to_sample_ratio / self.image_inv.shape[0] 
            h5dataset.attrs['element_size_um'] =  [z_sample_period,0.65,0.65]


        finally:
            parent.close()
        
        
#%%    
 
if __name__ == "__main__" :
    

        #   ----  1 ----- 
        # file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/data_28_4_22/220428_124841_coherent_SVIM_phantom2_good.h5'
        #   ----  2 ----- 
        # file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/data_28_4_22/220428_124222_coherent_SVIM_phantom2_good.h5'
        #   ----  3 ----- 
        # file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/data_28_4_22/220428_125643_coherent_SVIM_phantom2_good.h5'
        #   ----  4 ----- 1
        # file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/data_28_4_22/220428_115143_coherent_SVIM_phantom2_good.h5'
        
        
        # ------ with diffurer
        file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/220509_gfp_plant/220509_163108_coherent_SVIM_plant1_elongation_diffuser.h5'
        # ------ Diffuser off
        # file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/220509_gfp_plant/220509_163256_coherent_SVIM_plant1_elongation_diffuser_stoppeed.h5'
        # ------ No diffuser
        # file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/220509_gfp_plant/220509_163441_coherent_SVIM_plant1_elongation_no_diffuser.h5'
        # ------ Camaleon
        # file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/220509_gfp_plant/220509_172152_coherent_SVIM_plant2.h5'
        
        
        # ------ plant in the center, left and right?, no diffuser
        # file_name = "D:\\data\\coherent_svim\\220509_gfp_plant\\220509_150543_coherent_SVIM_plant1.h5"
        # file_name = "D:\\data\\coherent_svim\\220509_gfp_plant\\220509_150951_coherent_SVIM_plant1.h5"
        # file_name = "D:\\data\\coherent_svim\\220509_gfp_plant\\220509_151407_coherent_SVIM_plant1.h5"
        
        # ------ transposed pattern, there was the problem that the frequencies were a bit unclear and there was a divide by zero traceback
        # file_name = "D:\\data\\coherent_svim\\220509_gfp_plant\\220509_151641_coherent_SVIM_plant1_transp.h5"
        # file_name = "D:\\data\\coherent_svim\\220509_gfp_plant\\220509_151812_coherent_SVIM_plant1_transp.h5"
        # file_name = "D:\\data\\coherent_svim\\220509_gfp_plant\\220509_152405_coherent_SVIM_plant1_transp_20pz_z.h5"
        
        # ------ different exposure times
        # file_name = "D:\\data\\coherent_svim\\220509_gfp_plant\\220509_153249_coherent_SVIM_plant1_tip_200ms.h5"
        # file_name = "D:\\data\\coherent_svim\\220509_gfp_plant\\220509_153510_coherent_SVIM_plant1_tip_800ms.h5"
        
        
        # (220511) H2O2 stimulus
        
        # file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/220510_gfp_plant/renamed/pog_h2o2_1.h5'
        # file_name = "D:\\data\\coherent_svim\\220511_camaleon_plant\\220511_114841_coherent_SVIM_cy36_atp_second_round_ (7).h5"
        # file_name = "D:\\data\coherent_svim\\220511_camaleon_plant\\220511_131028_coherent_SVIM_cy36_atp_third_round.h5"
        # file_name = "D:\\data\coherent_svim\\220511_gfp_plant\\220511_145138_coherent_SVIM_ (90).h5"
        # file_name = "D:\\data\\coherent_svim\\220511_gfp_plant\\220511_155605_coherent_SVIM_rog_h2o2_second_round_ (54).h5"
        # file_name = "D:\\data\\coherent_svim\\220511_gfp_plant\\220511_173125_coherent_SVIM_rog_h2o2_leaf_ (1).h5"
        
        
        # file_name_h5 = file_name + '.h5'
        
        dataset = coherentSVIM_analysis(file_name)
        dataset.load_h5_file()
        
        
        dataset.merge_pos_neg()
        # dataset.setROI(814-20,  1132-20, 40) # one bead in dataset 4
        # dataset.setROI(420,  524, 1000)
        # dataset.setROI(1839-110, 879-110 , 220) # three beads in dataset 1
        
        
        # dataset.setROI(396, 469 , 201, 50)
        # dataset.setROI(196, 469 , 201)
        # dataset.setROI(200, 0 , 600, 1024)
        dataset.show_im_raw()
        
        dataset.choose_freq() # also removes any duplicate in frequency
        
        #%% invert the raw image
        
        base = 'cos'
        mu = 0.01
        lamda = 0.5
        niter_out = 15
        niter_in = 2
        lsqr_niter = 5
        lsqr_damp = 1e-4
        
        dataset.p_invert(base = base)
        
        # dataset.invert_and_denoise1D_no_for(base = base, lamda = lamda, niter_out = niter_out,
                                # niter_in = niter_in, lsqr_niter = lsqr_niter, lsqr_damp = lsqr_damp)
        # invert_and_denoise3D_v2(base = base, lamda = lamda, niter_out = niter_out,
                                # niter_in = niter_in, lsqr_niter = lsqr_niter, lsqr_damp = lsqr_damp)
        
        # dataset.show_inverted()
        
        # dataset.show_inverted_xy()
        # dataset.show_inverted_xz()
        
        #%% show a single line profile along the z direction
        

        fig, ax = plt.subplots()
        ax.plot(dataset.image_inv[:,20,20],'x-', label = f'{dataset.params}')
        ax.legend()
        
        
        
        #%% save the inverted image

        # save_file = 'Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/data_28_4_22/220428_124841_coherent_SVIM_phantom2_good_inverted.tif'        
        
        # try:
        #     os.remove(save_file)
        # except:
        #     print('file name not found')
            
            
        # dataset.save_inverted(save_file)
        
        try:
        
            fname = "D:\\LabPrograms\\ScopeFoundry_POLIMI\\smSVIM_microscope_analyser\\analysed\\220511\\volume_1_rog_leaf.h5"
            parent = h5py.File(fname,'w')
    
            # create groups
            results = parent.create_group('level1') 
            
    
            # create a dataset
    
            parent.create_dataset('inverted_image', shape=dataset.image_inv.shape, data = dataset.image_inv)
            # parent['voltage2'] = signalA
    
            # create attributes (not in the parents)
            results.attrs['sample'] = 'GFP'
            results.attrs['microscope_type'] = 'SIM'
            results.attrs['inversion_parameters'] = dataset.param

        finally:
            parent.close()
            
            
        #%% load a saved reconstructed image
        
        import h5py
        
        # import sys
        # import pyqtgraph as pg
        # import qtpy.QtCore
        # from qtpy.QtWidgets import QApplication
        
        # fname = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/220509_163108_coherent_SVIM_plant1_elongation_diffuser_INVERTED_denoise_05.h5'
        fname = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/220509_163108_coherent_SVIM_plant1_elongation_diffuser_INVERTED_2.h5'
        
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
        # they have been reanmed in a quick and dirty way using MS file explorer to have a sequential name
        
        # fname = "D:\\data\\coherent_svim\\220511_camaleon_plant\\220511_114841_coherent_SVIM_cy36_atp_second_round_ ("
        # fname = "D:\\data\\coherent_svim\\220511_camaleon_plant\\220511_131028_coherent_SVIM_cy36_atp_third_round_ ("
        
        # >> In the next dataset we can see a response to the external stimulus << 
        fname= "D:\\data\\coherent_svim\\220511_gfp_plant\\220511_155605_coherent_SVIM_rog_h2o2_first_round_ ("
        
        # fname= "D:\\data\\coherent_svim\\220511_gfp_plant\\220511_155605_coherent_SVIM_rog_h2o2_second_round_ ("
        # fname = "D:\\data\\coherent_svim\\220511_gfp_plant\\220511_173125_coherent_SVIM_rog_h2o2_leaf_ ("
        
        
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
            
        