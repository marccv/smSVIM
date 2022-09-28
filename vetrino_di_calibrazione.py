# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:28:42 2022

@author: SPIM-OPT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
plt.rcParams['figure.dpi'] = 300

fname = '/Users/marcovitali/Documents/Poli/tesi/coherentSVIM/data/220919_zebrafish_5day_20x_h2o/Values_detection.csv'

data = np.genfromtxt(fname, delimiter=',', skip_header = 1)

fig, ax = plt.subplots()

ax.plot(data[:,0], data[:,1])

peak_prominence = 2
peak_height = 3e3

peaks, prop = find_peaks(data[:,1], prominence = peak_prominence, height = peak_height)

ax.plot(data[peaks,0], data[peaks,1], 'x', markersize = 3)
ax.set_xlabel('Pixel')


#%%
ave_dist = np.mean(np.diff(data[peaks,0])) #px
print(f'Average peak distance: {ave_dist:.3f} px')

th_dist = 10e-6 #m

pixel_dim_at_sample = th_dist/ave_dist
print(f'Camera pixel dimention at sample: {pixel_dim_at_sample*1e6:.3f} um/px' )


pixel_dim_at_camera = 6.5e-6

print(f'Pixel dimention at camera: {pixel_dim_at_camera*1e6:.3f} um/px')

magnification = pixel_dim_at_camera/pixel_dim_at_sample

print(f'Magnification: {magnification:.3f}')
