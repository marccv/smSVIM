#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 12:02:20 2022

@author: Marco Tobia Vitali

-------------------------------------------------------------------------------

Analyser of H5 dataset acquired with a hadamard type base

Given:
    - file_path

Returns:
    - it saves the inverted volume H5 file in a new folder with name : "{file_path}_ANALYSED"
      The output H5 file is made to be easily opened with FIJI


With the inversion setting "make_posneg" one can choose to reconstruct the negative 
values of the raw images, see the function make_pos_neg for more informations.

"""

import numpy as np
from scipy.linalg import hadamard

import time
from numpy.linalg import lstsq

import h5py
import os






def time_it(method):
    """Fucntion decorator to time a methos""" 
       
    def inner(*args,**kwargs):
        
        start_time = time.time() 
        result = method(*args, **kwargs) 
        end_time = time.time()
        print(f'Execution time for method "{method.__name__}": {end_time-start_time:.6f} s \n') 
        return result        
    return inner
    




# =============================================================================
# Functions to import the H5 file
# =============================================================================

# Written by Andrea Bassi (Politecnico di Milano) 10 August 2018
# to find the location of datasets in a h5 file and to extact attributes.


def get_h5_dataset(fname, dataset_index=0):
        """
        Finds the datasets in HDF5 file.
        Returns the dataset specified by the dataset_index.
        """
        try:
            f = h5py.File(fname,'r')
            name,shape,found = _get_h5_dataset(f, name=[], shape=[], found=0)    
            #assert found > 0, "Specified h5 file does not exsist or have no datasets"    
            if dataset_index >= found:    
                dataset_index = 0
            data = np.single(f[name[dataset_index]])
        
        finally:
            f.close()
        return data

def _get_h5_dataset(g, name, shape, found) :
        """
        Extracts the dataset location (and its shape).
        It is operated recursively in the h5 file.
        """
       
        if isinstance(g,h5py.Dataset):   
            found += 1
            name.append(g.name)
            shape.append(g.shape)
            
        if isinstance(g, h5py.File) or isinstance(g, h5py.Group):
            for key,val in dict(g).items() :
                
                name,shape,found = _get_h5_dataset(val,name,shape,found)
                 
        return name,shape,found 
    

def get_h5_attr(fname, attr_name):
    """
    Finds an attribute in with the specified names, in a h5 file
    Returns a dictionary with name as key and the attribute value
    Raise a Warning if more that one attribute with the same name is found.
    
    """
    try:
        f = h5py.File(fname,'r')
        attr_dict, found = _get_h5_attr(f, attr_name, value=[], found=0)
        if found > 1:
            print(f'Warning: more than one attribute with name {attr_name} found in h5 file')
    finally:
        f.close()        
    return attr_dict  
        
        
def _get_h5_attr(g, attr_name='some_name', value=[], found=0):
    """
    Returns the attribute's key and value in a dictionary.
    It is operated recursively in the h5 file. 
    """
    if isinstance(g, h5py.File) or isinstance(g, h5py.Group):
        
        for key,val in dict(g).items() :
            for attr_key, attr_val in val.attrs.items():
                if attr_key == attr_name:
                    found +=1
                    value.append(attr_val)
            value, found = _get_h5_attr(val, attr_name, value, found)
    
    return value, found
                  

    




# =============================================================================
# Function to create the problem matrix
# =============================================================================

def create_hadamard_matrix(num_of_patterns, had_type = 'normal'):
    
    """
    Returns Hadamard matrix given the number of pattern and its type:
        
        - "normal"    for normal hadamard matrix
        - "walsh"     for Walsh Hadamard ordered patterns
        - "scrambled" for a row and colum permutation of the normal hadamard matrix
        
    The matrix generated has +1 and -1 elements, not the +1 and 0 elements that 
    we can generate on the DMD
    
    """
    
    
    if had_type == 'normal':
        return hadamard(num_of_patterns)
    
    elif had_type == 'walsh':
        
        H = hadamard(num_of_patterns)
        diffs = np.diff(H)
        norm = np.linalg.norm(diffs, axis = 1)
        order = np.argsort(norm)
         
        return H[order,:]
    
    elif had_type == 'scrambled':
        np.random.seed(222)
        
        I = np.eye(num_of_patterns)
        Pr = I[np.random.permutation(num_of_patterns), :]
        Pc = I[np.random.permutation(num_of_patterns), :]
        
        return Pr @ hadamard(num_of_patterns) @ Pc
    
    
    
    
    
# =============================================================================
# Functions to perform the inversion
# =============================================================================
    
    
@time_it
def load_h5_file(file_path, dataset_index = 0):
    """ 
    Returns the dataset with insex 'dataset_index' in the H5 file at ' file path' 
    
    """
    
    return get_h5_dataset(file_path, max(0,dataset_index))


@time_it
def merge_pos_neg(imageRaw):
    """
    Subtracts the pairs of positive and negative raw images 
    
    """
        
    num_frames = imageRaw.shape[0]
    
    pos = imageRaw[np.linspace(0, num_frames -2, int(num_frames/2), dtype = 'int'), :, :]
    neg = imageRaw[np.linspace(1, num_frames -1, int(num_frames/2), dtype = 'int'), :, :]
    
    return pos - neg
        
@time_it
def make_pos_neg(imageRaw):
    """
    Use this function the raw images were not pairs of positive and negative
    
    WARNING >> SO FAR IT ONLY WORK FOR FREQUENCY ORDERED WALSH PATTERNS <<
    
    It subtracts the average of the high frequency pattern signals (empirically 
    from the number 7 on) to reconstruct the negative values.
    
    This allows us to consider the hadamard matrix with +1 and -1 entries as the
    problem matrix. Otherwise for only positive aquisition the matrix has +1 and 0
    entries: this is a constraint imposed by the DMD.

    """
    
    
    return 2*(imageRaw - np.mean(imageRaw[7:,:,:], 0))
    


@time_it
def lsqr_invert(imageRaw, base, make_posneg, file_path, dark_counts):
    
    '''
    Inverts the raw image using the least squares method
    
    '''

    
    had_pat_num= int(get_h5_attr(file_path, 'had_pat_num')[0]) # We read in the dataset the matrix dimension
    PosNeg = get_h5_attr(file_path, 'PosNeg')[0] # We read in the dataset if the dataset was Positive and Negative
    
    
    # Probelm matrix
    matrix = create_hadamard_matrix(had_pat_num, base)
        
    
    # If the acquisition is not PosNeg or we do not reconstruct the negative values 
    # we need to make the matrix with only +1 and 0 (as the real patterns on the DMD)
    if not PosNeg and not make_posneg:
        matrix[matrix <0] = 0
        
        # subtract the dark counts
        imageRaw -= dark_counts




    matrix = matrix.astype(float)
    Nz = matrix.shape[1]
    nz,ny,nx = imageRaw.shape
    
    image_inv,_,_,_ = lstsq(matrix, imageRaw.reshape( nz, int(ny*nx)), rcond = None)
    
    
    return image_inv.reshape(Nz,ny,nx)
    












        
# =============================================================================
# Main script
# =============================================================================
   
    
if __name__ == "__main__" :
    
    
    
    # Experimental setup constants
    
    dmd_to_sample_ratio = 1.195 #(um/px)
    pixel_size = 0.339 #(um/px)
    dark_counts = 99.6
    
    
    
    
    # Insert here the file path of the raw H5 dataset
    file_path = ""
    
    
    
    # --- Inversion settings ---
    
    
    # This "make_posneg" setting works on datasets that do not have the subsequent
    # positive and negative frames.
    # 
    # If True the raw image is processed to reconstruct its negative values (and 
    # thereafter use in the inverse problem the Hadamard matrix with +1 and -1 values
    # and not only +1 and 0 entries)
    
    make_posneg = False
    
    #----------------------------
    
    
    
    
    
    
    # read info saved in the H5 dataset
    base = get_h5_attr(file_path, 'had_type')[0]
    PosNeg = get_h5_attr(file_path, 'PosNeg')[0]
    ROI_s_z = get_h5_attr(file_path, 'ROI_s_z')[0]
    
    try:
        try:
            time_frames_n= get_h5_attr(file_path, 'real_time_frames_n')[0]
        except:
            time_frames_n = get_h5_attr(file_path, 'time_frames_n')[0]
    except:
        print('>> Warning: Could not find the number of time frames.')
        
    else: # if there are no errors
    
    
        try:
            
            # saves all outputs in a new folder
            # The output H5 file is made to be easily opened with FIJI
            
            head, tail = os.path.split(file_path)
            
            newpath = file_path[:-3] + '_ANALYSED'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            
            
            fname = os.path.join(newpath, 'time_lapse_inverted_complete.h5')
            
            while os.path.exists(fname):
                fname = fname[:-3] + '_bis.h5'
            
            parent = h5py.File(fname,'w')
    
    
    
            # Loop to invert all the time frames
            
            for time_index in range(time_frames_n):
                
                imageRaw = load_h5_file(file_path, time_index)
                
                if PosNeg : 
                    imageRaw = merge_pos_neg(imageRaw)
                elif make_posneg:
                    imageRaw = make_pos_neg(imageRaw)
        
        
        
                image_inv = lsqr_invert(imageRaw, base, make_posneg, file_path, dark_counts)
                
                    
                # create a dataset
                name = f't{time_index:04d}/c0000/image'
                h5dataset = parent.create_dataset(name = name, shape=image_inv.shape, data = image_inv)
                h5dataset.dims[0].label = "z"
                h5dataset.dims[1].label = "y"
                h5dataset.dims[2].label = "x"
                
                
                
                depth_z = (ROI_s_z * dmd_to_sample_ratio/ image_inv.shape[0] )
                h5dataset.attrs['element_size_um'] =  [depth_z, pixel_size, pixel_size]
            
        finally:
            parent.close()
    
    
    
    
    
    
    
    
    