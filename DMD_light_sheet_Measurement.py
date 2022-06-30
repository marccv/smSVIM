# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:23:56 2022

@author: SPIM-OPT
"""



from base_SVIM_Measurement import BaseSvimMeasurement
import numpy as np
        


def create_light_sheets(ROI_size = [270, 810], sheet_width = 10, transpose = False,  im_size = [1080, 1920]):
    
    """
    ROI_size:           dimentions of the rectangle ROI. First dimention refers to the border
                        parallel to the diagonal direction and the second to the border parallel to the
                        anti-diagonal direction. Default is [270,810], the largest rectangle with
                        borders proportioned 3:1 in a 1080x1920 image
    sheet_width:        width of the single light sheet rectangle
    transpose:          False -> the different sheets are parallel to the anti-diagonal direction
                        True  -> the different sheets are parallel to the diagonal direction
    im_size:            size of the whole image
    """
   
    
    s_y = im_size[0]
    s_x = im_size[1]
    
    # dimentions of the rectangle to be cropped out in units of pizel diagonal (same as unit_period)
    s_diag = ROI_size[0] #dimension of the border parallel to the diagonal direction
    s_anti = ROI_size[1] #dimension of the border parallel to the antidiagonal direction  

    delta_x_0 = - s_x/2 + s_anti/2 + s_diag/2
    delta_y_0 = - s_y/2 + s_anti/2 - s_diag/2
    
    if s_diag + s_anti > np.min(im_size):
        print('WARNING: The cropped field rectangle exceeds the DMD')
   
    X,Y = np.meshgrid(range(s_x), range(s_y))
    
    # @time_it
    def rectangle(x,y, s_diag, s_anti):
       return np.multiply( np.multiply(-1< x+y , x + y < 2*s_anti)  , np.multiply(  -1< x-y , x-y  < 2*s_diag))
    
    sheets = []
   
    if not transpose:
        if s_diag % sheet_width != 0:
            print('The ROI is not a multiple of light sheet width')
        N = int(np.floor(s_diag/sheet_width))
        
        delta_x = np.linspace(0, N*sheet_width, N, dtype = int, endpoint = False)
        delta_y = -delta_x
        
        
        for i in range(N):
            sheet = rectangle((X + delta_x_0 - delta_x[i]),  (Y + delta_y_0 - delta_y[i]), sheet_width, s_anti  )*np.uint8(1)
            sheets.append(sheet)
        
    else:
        if s_anti % sheet_width != 0:
            print('The ROI is not a multiple of light sheet width')
        N = int(np.floor(s_anti/sheet_width))
        
        delta_x = np.linspace(0, N*sheet_width, N, dtype = int, endpoint = False)
        delta_y = delta_x
         
        for i in range(N):
            sheet = rectangle((X + delta_x_0 - delta_x[i]),  (Y + delta_y_0 - delta_y[i]), s_diag , sheet_width )*np.uint8(1)
            sheets.append(sheet)
    
    return sheets





class DMD_light_sheet_measurement(BaseSvimMeasurement):     
    
    name = "DMD_light_sheet"
    
    def calculate_num_frames(self):
        if not self.settings['transpose_pattern']:
             return (1 + self.settings['PosNeg']) * int(np.floor(self.settings['ROI_s_z']/self.settings['sheet_width']))
        else:
             return (1 + self.settings['PosNeg']) * int(np.floor(self.settings['ROI_s_y']/self.settings['sheet_width']))
        
    
    def set_sheet_width(self, sheet_width):
        
        if hasattr(self, 'transpose_pattern'):
            self.settings['num_frames'] = self.calculate_num_frames()
               
        if hasattr(self, 'time_frames_n'):
            self.settings['time_frames_n'] = self.calculate_time_frames_n()



    def setup_svim_mode_settings(self):
        

        self.sheet_width = self.settings.New('sheet_width', dtype = int, initial = 10, vmin = 1, unit = 'px')
        
        #set functions
        self.sheet_width.hardware_set_func = self.set_sheet_width
        
           
    def run_svim_mode_function(self):
            
            transpose_pattern = self.settings['transpose_pattern']
            ROI_size = [self.settings['ROI_s_z'], self.settings['ROI_s_y']]
            
            
            if self.settings['PosNeg'] == False:
                 
                images = create_light_sheets(ROI_size , self.settings['sheet_width'], transpose_pattern)
                
            else:
                #PosNeg
                images = []
                im_pos = create_light_sheets(ROI_size , self.settings['sheet_width'], transpose_pattern)
                
                
                for im in im_pos:
                    images.append(im)
                    im_neg = np.uint8(np.logical_not(im)*1)
                    images.append(im_neg)
                        
            return images