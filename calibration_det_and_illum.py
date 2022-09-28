# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:28:42 2022

@author: SPIM-OPT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
plt.rcParams['figure.dpi'] = 300

#%%

# DETECTION
print('--------------------------------------\n\n> DETECTION\n')

fname = '/Users/marcovitali/Documents/Poli/tesi/coherentSVIM/data/220919_zebrafish_5day_20x_h2o/Values_detection.csv'

data = np.genfromtxt(fname, delimiter=',', skip_header = 1)

fig, ax = plt.subplots()
fig.suptitle('DETECTION')

ax.plot(data[:,0], data[:,1])

peak_prominence = 2
peak_height = 3e3

peaks, prop = find_peaks(data[:,1], prominence = peak_prominence, height = peak_height)

ax.plot(data[peaks,0], data[peaks,1], 'x', markersize = 5)
ax.set_xlabel('Pixel')


ave_dist = np.mean(np.diff(data[peaks,0])) #px
print(f'Average peak distance ({len(peaks)} peaks found): {ave_dist:.3f} px')

th_dist = 10e-6 #m
print(f'Theoretical peak distance: {th_dist*1e6:.3f} um' )



pixel_dim_at_sample = th_dist/ave_dist
print(f'Camera pixel dimension at sample: {pixel_dim_at_sample*1e6:.3f} um/px' )


pixel_dim_at_camera = 6.5e-6

print(f'Pixel dimension at camera: {pixel_dim_at_camera*1e6:.3f} um/px')

magnification = pixel_dim_at_camera/pixel_dim_at_sample

print(f'Magnification: {magnification:.3f}')



#%%  ILLUMINATION

print('\n\n> ILLUMINATION\n')

# fname = '/Users/marcovitali/Documents/Poli/tesi/coherentSVIM/data/220919_zebrafish_5day_20x_h2o/Values_illumination.csv'

fname = '/Users/marcovitali/Downloads/Values_2.csv'
pixel_dim_at_sample = 0.339e-6#m


data = np.genfromtxt(fname, delimiter=',', skip_header = 1)

fig, ax = plt.subplots()
fig.suptitle('ILLUMINATION')

ax.plot(data[:,0], data[:,1])

peak_prominence = 2
peak_height = 180

peaks, prop = find_peaks(data[:,1], prominence = peak_prominence, height = peak_height)

ax.plot(data[peaks,0], data[peaks,1], 'x', markersize = 5)
ax.set_xlabel('Pixel')



ave_dist = np.mean(np.diff(data[peaks,0])) * pixel_dim_at_sample #m
print(f'Average peak distance ({len(peaks)} peaks found): {ave_dist*1e6:.3f} um')


modulation_period_on_dmd = 5 #px
dmd_pixel_size = 7.56e-6*np.sqrt(2) #m
print(f'Diagonal DMD pixel dimension: {dmd_pixel_size*1e6:.3f} um')


dmd_pixel_at_sample = ave_dist/modulation_period_on_dmd
print(f'DMD pixel on the sample: {dmd_pixel_at_sample*1e6:.3f} um')


magnification = dmd_pixel_size*modulation_period_on_dmd/ave_dist

print(f'Magnification: {magnification:.3f}\n--------------------------------------')
