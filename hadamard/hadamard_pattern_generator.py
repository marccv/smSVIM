#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:07:26 2022

@author: marcovitali
"""

import numpy as np
from scipy.linalg import hadamard
import matplotlib.pyplot as plt




def scramble(H, N):
    
    np.random.seed(222)
    
    I = np.eye(N)
    Pr = I[np.random.permutation(N), :]
    Pc = I[np.random.permutation(N), :]
    return Pr @ H @ Pc


def walsh_gen(n):
    
    H = hadamard(n)
    diffs = np.diff(H)
    norm = np.linalg.norm(diffs, axis = 1)
    order = np.argsort(norm)
     
    return H[order,:]







def create_hadamard_patterns(num_of_patterns = 32, had_type = 'normal' , transpose_pattern=False, cropped_field_size = [256, 600],
                             im_size = [1080, 1920]):
    
    s_y = im_size[0]
    s_x = im_size[1]
    
    # dimentions of the rectangle to be cropped out in units of pizel diagonal (same as unit_period)
    s_diag = cropped_field_size[0] #dimension of the border parallel to the diagonal direction
    s_anti = cropped_field_size[1] #dimension of the border parallel to the antidiagonal direction  
    
    
    if had_type == 'normal':
        H = hadamard(num_of_patterns)
    elif had_type == 'walsh':
        H = walsh_gen(num_of_patterns)
    elif had_type == 'scrambled':
        H = scramble(hadamard(num_of_patterns), num_of_patterns)

    
    
    H[H<0] = 0 # the DMD only accepts 0 and 1, so to create the real pattern I will have to operate in PosNeg mode
    
    images = []
    
    if not transpose_pattern:
        #antidiag
        
        repetitions = s_diag/num_of_patterns * 2
        
        for i in range(num_of_patterns):
            
            image = np.zeros(im_size, dtype = 'uint8')
            
            strip = np.uint8(np.repeat(H[i], repetitions).reshape(1,s_diag*2).copy())
            
            pad_len = s_y + 0.5*(s_x - s_y - s_diag*2)
            padding = np.zeros([1,int(pad_len)])
            
            strip = np.concatenate((padding, strip, padding), axis = 1)
            
            # print(strip)
            
            for j in range(s_y):
                image[j, :] = strip[0, (s_y-j-1):(s_y + s_x -j-1)]
                
            images.append(image)
    else:
        #transpose: diag
        repetitions = s_anti/num_of_patterns * 2
        
        for i in range(num_of_patterns):
            
            image = np.zeros(im_size, dtype = 'uint8')
            
            strip = np.uint8(np.repeat(H[i], repetitions).reshape(1,s_anti * 2).copy())
            
            pad_len = s_y + 0.5*(s_x - s_y - s_anti * 2)
            padding = np.zeros([1,int(pad_len)])
            
            strip = np.concatenate((padding, strip, padding), axis = 1)
            
            # print(strip)
            
            for j in range(s_y):
                image[j, :] = strip[0, j :( s_x +j)]
                
            images.append(image)
        
            
    return images,H







def create_rectangle_mask(cropped_field_size = [256, 600], im_size = [1080, 1920]):
    
    """
    cropped_field_size: dimentions of the rectangle ROI. First dimention refers to the border
                        parallel to the diagonal direction and the second to the border parallel to the
                        anti-diagonal direction. Default is [270,810] to have the largest rectangle with
                        borders proportioned 3:1
    im_size           : Size of the outpun image in pizel; default is [1080,1920]
    """
   
    
    s_y = im_size[0]
    s_x = im_size[1]
    
    # dimentions of the rectangle to be cropped out in units of pizel diagonal (same as unit_period)
    s_diag = cropped_field_size[0] #dimension of the border parallel to the diagonal direction
    s_anti = cropped_field_size[1] #dimension of the border parallel to the antidiagonal direction  


    if s_diag + s_anti > np.min(im_size):
        print('WARNING: The cropped field rectangle exceeds the DMD')
        
    # @time_it
    def rectangle(x,y, s_diag, s_anti):
       return np.multiply( np.multiply(-s_anti -1< x+y , x + y < s_anti)  , np.multiply( -s_diag -1< x-y , x-y  < s_diag))

    X,Y = np.meshgrid(range(s_x), range(s_y))
    mask = rectangle((X - s_x/2 ) , (Y - s_y/2) , s_diag, s_anti)
    
    
    return mask
    
    
            
            
if __name__ == '__main__':
    
    n = 16
    # had_type = 'normal'
    had_type = 'walsh'
    # had_type = 'scrambled'
    
    images, H = create_hadamard_patterns(n, had_type, transpose_pattern=False)
    
    print(H)
    fig1=plt.figure()
    fig1.clf()
    # fig1.suptitle(f'Inverted image XY projection\n{self.params}')
    ax1=fig1.add_subplot(111)
    xy = ax1.imshow(H)
    
    fig1.colorbar(xy, ax=ax1)
    plt.title(f'{had_type} Hadamard, n = {n}')
    
    
    mask = create_rectangle_mask()
    images = images*mask
    
    
    for image in images:
        
    
        fig1=plt.figure()
        fig1.clf()
        # fig1.suptitle(f'Inverted image XY projection\n{self.params}')
        ax1=fig1.add_subplot(111)
        
        
        # I add this white lines to show the center of the frame
        image[:, int(1920/2)] = 0
        image[int(1080/2), :] = 0
        
        xy = ax1.imshow(image)
        
        fig1.colorbar(xy, ax=ax1)
        
        
    # To show that the pattern is actually centered in the DMD frame
    # print(images[1,int(1080/2-15):int(1080/2+15),int(1920/2-15):int(1920/2+15)])
    
    
    # np.savetxt(f'{had_type}_had_{n}.csv', H , fmt='%.3f', delimiter = ', ')
    
    
    
    
    
    
    
    # #%%
    # n = 8
    # mat = hadamard(n)
    
    # b = 1/n * mat@mat
    
    # print(b)
    
    
    # #%%
    # s = 0
    # for i in range(1000):
    
    #     a = 3*np.random.random(n)
        
    #     # H[H == 0] = -1
        
    #     ave_illum = np.mean(H@a, 0)
    #     s += ave_illum
    
    # print(s/1000)
    
    
    