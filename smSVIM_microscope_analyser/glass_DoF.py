#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:47:56 2022

@author: marcovitali
"""

import numpy as np
from get_h5_data import get_h5_dataset, get_h5_attr
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 9})
import h5py



#%%

# basename = '/Users/marcovitali/Documents/Poli/tesi/coherentSVIM/data/220610_glass_holder_pi_stage/4b_fourth_no_glass/renamed/step_'
# basename = '/Users/marcovitali/Documents/Poli/tesi/coherentSVIM/data/220610_glass_holder_pi_stage/4b_fourth_no_glass/renamed/220610_124932_hamamatsu_image('


# basename = '/Users/marcovitali/Documents/Poli/tesi/coherentSVIM/data/220610_glass_holder_pi_stage/4a_fourth/renamed/step_'

basename = '/Users/marcovitali/Documents/Poli/tesi/coherentSVIM/data/220610_glass_holder_pi_stage/3a_third_ERROR_DIFF_OFF/renamed/step_'

stack = []

# fname = '/Users/marcovitali/Documents/Poli/tesi/coherentSVIM/data/220610_glass_holder_pi_stage/4b_fourth_no_glass/merged.h5'
# fname = '/Users/marcovitali/Documents/Poli/tesi/coherentSVIM/data/220610_glass_holder_pi_stage/4a_fourth/merged.h5'

fname = '/Users/marcovitali/Documents/Poli/tesi/coherentSVIM/data/220610_glass_holder_pi_stage/3a_third_ERROR_DIFF_OFF/merged.h5'


parent = h5py.File(fname,'w')


try:

    for i in range(36):
        
        name = basename + str(i) + '.h5'
        
        # name = basename + str(i+1) + ').h5'
        
        step = get_h5_dataset(name, 0)
        stack.append(step)
        
        name = f't{i:02d}/c00'
    
    
        h5dataset = parent.create_dataset(name, shape=step.shape, data = step)
    
        # h5dataset.dims[0].label = "t"
        # h5dataset.dims[1].label = "y"
        # h5dataset.dims[2].label = "x"
        
        h5dataset.attrs['element_size_um'] =  [10 ,0.65,0.65]

finally:
    parent.close()