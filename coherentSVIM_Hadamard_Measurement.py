# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:54:03 2022

@author: SPIM-OPT
 
   Written by Michele Castriotta, Alessandro Zecchi, Andrea Bassi (Polimi).
   Code for creating the measurement class of ScopeFoundry for the Orca Flash 4V3
   11/18
"""

from base_SVIM_Measurement import BaseSvimMeasurement
import numpy as np
from scipy.linalg import hadamard



def scramble(H, N):
    
    # I do not set the seed beacuse for compressed sensing we want different scrambled hadamard patterns for each time point
    # I will save the displayed patterns in the H5 file
    #np.random.seed(222)
    
    I = np.eye(N)
    Pr = I[np.random.permutation(N), :]
    Pc = I[np.random.permutation(N), :]
    return Pr @ H @ Pc


# def walsh_gen_old(n):
    
#     from numpy import genfromtxt
#     return genfromtxt(f"D:\\LabPrograms\\ScopeFoundry_POLIMI\\smSVIM_Microscope\\pattern\\walsh_hadamard\\wh{n}.csv", delimiter=',')

def walsh_gen(n):
    
    H = hadamard(n)
    diffs = np.diff(H)
    norm = np.linalg.norm(diffs, axis = 1)
    order = np.argsort(norm)
     
    return H[order,:]


def create_hadamard_patterns(num_of_patterns = 32, had_type = 'normal' , transpose_pattern=False, cropped_field_size = [256, 512],
                             im_size = [1080, 1920]):
    
    """
    had types: [ 'normal', 'walsh', 'scrambled']
    """
    
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

    H_posneg = H.copy()
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
        
            
    return images, H_posneg







class coherentSvim_Hadamard_Measurement(BaseSvimMeasurement):     
    
    name = "coherentSvim_Hadamard"
    
    def calculate_num_frames(self):
        # number of frames for a single volume
        
        # This on the other hand is the number of frames that the camera is going to acquire
        if not self.settings['comp_sensing']:
            n_frames = (1 + self.settings['PosNeg']) *  self.settings['had_pat_num']
    
        else:
            n_frames = (1 + self.settings['PosNeg']) *  self.settings['had_pat_num'] / self.settings['comp_factor']
          
            
        # This tells how big is the complete basis that we upload, at the begining of the measurement, to the DMD
        # The following expression is for the case of scrambled hadamard pattern with different seed for each time point
        
        if hasattr(self, 'time_point_num'):
            self.load_num_frames = int(n_frames * self.settings['time_point_num']) # must be <= 400
            print(f'load num frames = {self.load_num_frames}')
        return n_frames
    
    def set_had_pat_num(self, had_pat_num):
        
        if np.log2(had_pat_num)%1 != 0:
            
            higher = int(2**(np.ceil(np.log2(had_pat_num))))
            lower = int(higher/2)
            
            if had_pat_num ==  higher - 1:
                self.settings['had_pat_num'] = lower
            else:
                self.settings['had_pat_num'] = higher
        else:
            self.settings['num_frames']  = self.calculate_num_frames()
            
            if hasattr(self, 'time_point_num'):
                self.check_time_point_number()
            
            if hasattr(self, 'est_obs_time'):
                self.settings['est_obs_time'] = self.calculate_time_lapse_duration()
            
         
    def set_comp_sensing(self, comp_sensing):
        
        # this is overwritten in the particular measurement
        
        self.settings['num_frames'] = self.calculate_num_frames()
        self.settings['had_type'] = 'scrambled'
        
        if hasattr(self, 'time_point_num'):
                self.check_time_point_number()
    
        
    def set_comp_factor(self, comp_factor):
        
        # if cs_subset_dim > self.settings['had_pat_num']:
        #     self.settings['cs_subset_dim'] = self.settings['had_pat_num']
            
        # else:
        #     self.settings['num_frames'] = self.calculate_num_frames()
            
        if np.log2(comp_factor)%1 != 0:
            
            higher = int(2**(np.ceil(np.log2(comp_factor))))
            lower = int(higher/2)
            
            if comp_factor ==  higher - 1:
                self.settings['comp_factor'] = lower
            else:
                self.settings['comp_factor'] = higher
                
        else:
            self.settings['num_frames']  = self.calculate_num_frames()
            
            if hasattr(self, 'time_point_num'):
                self.check_time_point_number()
            
            if hasattr(self, 'est_obs_time'):
                self.settings['est_obs_time'] = self.calculate_time_lapse_duration()
        
    def setup_svim_mode_settings(self):
        
        self.had_pat_num = self.settings.New('had_pat_num', dtype=int, initial=16 , vmin = 1 )
        self.had_pat_num.hardware_set_func = self.set_had_pat_num
        
        
        self.settings.New('had_type', dtype = str, choices = [ 'normal', 'walsh', 'scrambled'], initial = 'scrambled')
        
        
        
    def run_svim_mode_function(self):
        transpose_pattern = self.settings['transpose_pattern']
        cropped_field_size = [self.settings['ROI_s_z'], self.settings['ROI_s_y']]
            
        
        images = []
        self.used_posneg_patterns =[]
        
        for time_index in range(self.settings['time_point_num']):
            print(f'Creating patterns for timepoint {time_index}')
            
            if self.settings['PosNeg'] == False:
                  
                time_point_images, H_posneg = create_hadamard_patterns( self.settings['had_pat_num'], self.settings['had_type'], transpose_pattern, cropped_field_size )
            
            else:
                #PosNeg
                im_pos, H_posneg = create_hadamard_patterns( self.settings['had_pat_num'], self.settings['had_type'], transpose_pattern, cropped_field_size )
                time_point_images =[]
                
                for im in im_pos:
                    time_point_images.append(im)
                    im_neg = np.uint8(np.logical_not(im)*1)
                    time_point_images.append(im_neg)
                    
            # I select the first (had_pat_num/comp_factor) out of the complete scrambled hadamard matrices    
            
            for image in time_point_images[0:self.settings['num_frames']]:
                images.append(image)
            
            if self.settings['comp_sensing'] == True:
                self.used_posneg_patterns.append(H_posneg[0:int(self.settings['had_pat_num']/self.settings['comp_factor']),:]) # here I cannot use directly num_frames as in the line before because of the posneg factor that can vary
            else:
                self.used_posneg_patterns.append(H_posneg)
                
        return images
        
    def run_iteration_cs_sequence(self, time_index):
        
        """
        Defines the basis subset (and its order) for compressed sensing acquisitions
        Note: the function must return a list
        """
    
        # returns a random list of length (cs_subset_dim) of indexes going from 0 to (had_pat_num - 1)
        # return list(np.random.permutation(self.settings['had_pat_num'])[0:self.settings['cs_subset_dim']])
        
        return list(np.linspace(start = time_index*self.settings['num_frames'], stop = (time_index + 1)*self.settings['num_frames'] -1, num = self.settings['num_frames'], dtype = int))
         