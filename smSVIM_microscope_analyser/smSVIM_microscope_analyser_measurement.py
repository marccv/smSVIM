# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:48:46 2022

@author: SPIM-OPT
"""

import numpy as np
import scipy.fftpack as sc_fft
import h5py
import numpy as np
from smSVIM_microscope_analyser.get_h5_data import get_h5_dataset, get_h5_attr
import time


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
    
    
    
    def __init__(self, fname):
        
        self.filename  = fname
        
        
    def load_h5_file(self,filename):
            self.imageRaw = get_h5_dataset(filename) #TODO read h5 info
            self.enableROIselection()
            
            #load some settings, if present in the h5 file
            for key in ['exposure', 'transpose_pattern','ROI_s_z','ROI_s_y']:
                val = get_h5_attr(filename, key)
                if len(val)>0:
                    new_value = val[0]
                    self.settings[key] = new_value
                    self.show_text(f'Updated {key} to: {new_value} ')
    
        
        
    
    
"""The following is only to test the functions.
"""     
if __name__ == "__main__" :
    
        import sys
        import pyqtgraph as pg
        import qtpy.QtCore
        from qtpy.QtWidgets import QApplication
        
        # this h5 file must contain a dataset composed by an array or an image
        file_name="D:\\LabPrograms\\ScopeFoundry_POLIMI\\smSVIM_Microscope\\data\\220422_124826_coherent_SVIM.h5"
        
        dataset = coherentSVIM_analysis(file_name)