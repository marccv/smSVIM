# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:25:23 2022

@author: SPIM-OPT
"""

import numpy as np
import transform_6090 as t_6090
import scipy.fftpack as sp_fft
from get_h5_data import get_h5_dataset, get_h5_attr
import tifffile as tiff
import os


import sys
import pyqtgraph as pg
import qtpy.QtCore
from qtpy.QtWidgets import QApplication

# file_name="D:\\LabPrograms\\ScopeFoundry_POLIMI\\smSVIM_Microscope\\data\\220422_124826_coherent_SVIM.h5"

class coherentSVIM_analysis:
    
    name = 'coherentSVIM_analysis'
    
    # disp_freq = np.array([0, 0.5, 1.0, 1.5037593984962405, 2.0, 2.5, 3.0303030303030303, 3.508771929824561, 4.0, 4.545454545454546, 5.0, 5.555555555555555, 6.0606060606060606, 6.666666666666667, 7.142857142857143, 7.6923076923076925, 8.0, 8.695652173913043, 9.090909090909092, 9.523809523809524, 10.0, 10.526315789473685, 11.11111111111111, 11.764705882352942, 12.5, 12.5, 13.333333333333334, 14.285714285714286, 14.285714285714286, 15.384615384615385, 15.384615384615385, 16.666666666666668, 16.666666666666668, 16.666666666666668, 18.181818181818183, 18.181818181818183, 18.181818181818183, 20.0, 20.0, 20.0, 20.0, 22.22222222222222, 22.22222222222222, 22.22222222222222, 22.22222222222222, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 28.571428571428573, 33.333333333333336, 33.333333333333336, 33.333333333333336])
    
    
    
    def __init__(self, fname):
        
        self.filename  = fname
        
        
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
    
    def setROI(self, x_min, y_min, ROIsize):
        
        self.imageRaw = self.imageRaw[:, x_min :x_min +ROIsize, y_min: y_min+ROIsize]
        
    def chose_freq(self, N):
        
        f_start = get_h5_attr(self.filename, 'f_min')[0]
        f_stop = get_h5_attr(self.filename, 'f_max')[0]
        ROI_s_z = get_h5_attr(self.filename, 'ROI_s_z')[0]
        
        freqs = np.linspace(f_start, f_stop, 2*(f_stop - f_start) + 1,dtype = float)
        disp_freqs = [0]
        
        for freq in freqs[1:]:
            period = int(ROI_s_z/freq)
            disp_freqs.append(ROI_s_z/period)
        
        
        self.imageRaw = self.imageRaw[0:N, :, :]
        self.disp_freqs = disp_freqs[0:N]
        
        
        # mask = np.append(np.logical_not(np.diff(disp_freqs)== 0), True)
        
        # self.imageRaw = self.imageRaw[mask, :, :]
        # self.disp_freqs = np.array(disp_freqs)[mask]
    
        
    def invert(self, base = 'sq'):
        
        self.transform = t_6090.dct_6090(self.disp_freqs)
        self.transform.create_space()
        
        if base == 'cos':
            self.transform.create_matrix_cos()
        elif base == 'sq':
            self.transform.create_matrix_sq()
        elif base == 'sp_dct':
            self.image_inv = sp_fft.idct(self.imageRaw, type = 2, axis = 0)
            return
        
        self.transform.compute_inverse()
        
        self.image_inv = np.tensordot(self.transform.inv_matrix ,  self.imageRaw , axes=([1],[0]))
        
    def show_inverted(self):
        pg.image(self.image_inv, title="Inverted image")        
               
        #keeps the window open running a QT application
        if sys.flags.interactive != 1 or not hasattr(qtpy.QtCore, 'PYQT_VERSION'):
            QApplication.exec_()
                          
   
        sys.exit ( "End of test")
    
    def save_inverted(self, outputFile):
        
        
        tiff.imsave(outputFile , np.uint16(self.image_inv.clip(min = 0)), append = False) 
        
    
 
if __name__ == "__main__" :
    

        
        # file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/220422_174325_coherent_SVIM.h5'
        file_name = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/data_22_4/220422_180920_coherent_SVIM_phantom1.h5'
        
        # file_name_h5 = file_name + '.h5'
        
        dataset = coherentSVIM_analysis(file_name)
        dataset.load_h5_file()
        # dataset.setROI(594,  306, 963)
        # dataset.show_im_raw()
        
        dataset.chose_freq(24)
        dataset.invert(base = 'sq')
        # dataset.show_inverted()
        
        
        
        # save_file = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/220422_174325_coherent_SVIM_inverted.tif'
        save_file = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/data_22_4/220422_180920_coherent_SVIM_phantom1_inverted.tif'        
        
        try:
            os.remove(save_file)
        except:
            print('file name not found')
            
            
        dataset.save_inverted(save_file)
        
        
        
        
        
        