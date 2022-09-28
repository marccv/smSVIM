#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:57:57 2022

@author: marcovitali
"""
import h5py

# =============================================================================
# Functions to import the H5 file
# =============================================================================

# Written by Andrea Bassi (Politecnico di Milano) 10 August 2018
# to find the location of datasets in a h5 file and to extact attributes.


    

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
                  


#---------------------------------------------------------------


fname = ''

used_patterns = get_h5_attr(fname, 'used_patterns')

print(used_patterns)