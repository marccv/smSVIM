# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:23:56 2022

@author: SPIM-OPT
"""



from ScopeFoundry import Measurement
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file
from ScopeFoundry import h5_io
import pyqtgraph as pg
import numpy as np
import os
import time
        


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

def create_rectangle_mask(cropped_field_size = [270, 810], im_size = [1080, 1920]):
    
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

class DMD_light_sheet(Measurement):     
    
    name = "DMD_light_sheet"
    

    def calculate_time_frames_n(self):
        if not self.settings['time_lapse']:
            return int(1)
        else:
            # 0.3s is the trigger dead time we set. 0.24 is an empirical time found to take in consideration computational dead times of each iteration of the for loop
            return int(np.ceil(    self.settings['obs_time'] / ( (self.settings['num_frames']/self.settings['effective_fps']) + self.settings['dark_time'] + 0.3 + 0.24 )    ))  
     
    
    
    def set_sheet_width(self, sheet_width):
        
        if hasattr(self, 'transpose_pattern'):
            if not self.settings['transpose_pattern']:
                 self.settings['num_frames'] = (1 + self.settings['PosNeg']) * int(np.floor(self.settings['ROI_s_z']/sheet_width))
            else:
                 self.settings['num_frames'] = (1 + self.settings['PosNeg']) * int(np.floor(self.settings['ROI_s_y']/sheet_width))
               
        if hasattr(self, 'time_frames_n'):
            self.settings['time_frames_n'] = self.calculate_time_frames_n()
            
    def set_PosNeg(self, PosNeg):
        
        if hasattr(self, 'num_frames'):
            
            if not self.settings['transpose_pattern']:
                self.settings['num_frames'] = (1 + PosNeg) * int(np.floor(self.settings['ROI_s_z']/self.settings['sheet_width']))
            else:
                self.settings['num_frames'] = (1 + PosNeg) * int(np.floor(self.settings['ROI_s_y']/self.settings['sheet_width']))
        
        if hasattr(self, 'time_frames_n'):
                self.settings['time_frames_n'] = self.calculate_time_frames_n()
                
    def set_ROI_s_z(self, ROI_s_z):
        
        if hasattr(self, 'transpose_pattern') and not self.settings['transpose_pattern'] :
            self.settings['num_frames'] = (1 + self.settings['PosNeg']) * int(np.floor(ROI_s_z/self.settings['sheet_width']))
        if hasattr(self, 'time_frames_n'):
                self.settings['time_frames_n'] = self.calculate_time_frames_n()
            
    def set_ROI_s_y(self, ROI_s_y):
        
        if hasattr(self, 'transpose_pattern') and self.settings['transpose_pattern'] :
            self.settings['num_frames'] = (1 + self.settings['PosNeg']) * int(np.floor(ROI_s_y/self.settings['sheet_width']))        
        if hasattr(self, 'time_frames_n'):
            self.settings['time_frames_n'] = self.calculate_time_frames_n()
            
    def set_transpose_pattern(self, transpose):
        
        if not transpose:
            self.settings['num_frames'] = (1 + self.settings['PosNeg']) * int(np.floor(self.settings['ROI_s_z']/self.settings['sheet_width']))
        else:
            self.settings['num_frames'] = (1 + self.settings['PosNeg']) * int(np.floor(self.settings['ROI_s_y']/self.settings['sheet_width']))
            
        if hasattr(self, 'time_frames_n'):
                self.settings['time_frames_n'] = self.calculate_time_frames_n()    
        
    def calculate_margin(self):
        
        read_one_line = 9.74436 #(us)
        delay =  9 * read_one_line #(us)
        contingency = 0.008
        self.eff_subarrayv = int(self.camera.subarrayv.val/self.camera.binning.val)
        
        return (1 + contingency) * (delay + (self.eff_subarrayv/2) * read_one_line) *1e-3 #(ms)

    def calculate_eff_fps(self):
        
        return 1000/(self.settings['exposure']+ self.settings['edge_trigger_margin']) #(fps)
     
    
    def set_exposure(self, exposure):
        
        self.camera.settings['exposure_time'] = exposure*1e-3
        self.settings['effective_fps'] =  self.calculate_eff_fps()
        
        if hasattr(self, 'time_frames_n'):
            self.settings['time_frames_n'] = self.calculate_time_frames_n()

    def read_subarray_vsize(self):
        
        self.settings['edge_trigger_margin'] = self.calculate_margin()
        self.settings['effective_fps'] =  self.calculate_eff_fps()     
        
        if hasattr(self, 'time_frames_n'):
            self.settings['time_frames_n'] = self.calculate_time_frames_n()
    
    def set_time_lapse(self, time_lapse):
        if hasattr(self, 'time_frames_n'):
            self.settings['time_frames_n'] = self.calculate_time_frames_n()
    
    def set_obs_time(self, obs_time):
        
        if hasattr(self, 'time_frames_n'):
            self.settings['time_frames_n'] = self.calculate_time_frames_n()
    def set_dark_time(self, dark_time):
        
        if hasattr(self, 'time_frames_n'):
            self.settings['time_frames_n'] = self.calculate_time_frames_n()
        
        
        
        
    def setup(self):
        
        "..."

        self.ui_filename = sibling_path(__file__, "coherentSVIM.ui")
        
        self.camera = self.app.hardware['HamamatsuHardware']
        self.dmd_hw = self.app.hardware['TexasInstrumentsDmdHW']
        self.shutter_hw = self.app.hardware['Shutter']
    
        self.ui = load_qt_ui_file(self.ui_filename)
        #self.settings.New('save_h5', dtype=bool, initial=False )
        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals=4, initial=0.04, vmin=0)
        self.settings.New('auto_range', dtype=bool, initial=True )
        self.settings.New('auto_levels', dtype=bool, initial=True )
        self.settings.New('level_min', dtype=int, initial=60 )
        self.settings.New('level_max', dtype=int, initial=150 )
        self.sheet_width = self.settings.New('sheet_width', dtype = int, initial = 10, vmin = 1, unit = 'px')
        self.PosNeg = self.settings.New('PosNeg', dtype = bool, initial = False)
        self.num_frames = self.settings.New('num_frames',ro = True, dtype = int, initial = 20)  # TODO The initial value of this setting is critical: so far it must be updated manually if one changes any other initial value. Should we calculate self.freqs during the setup period and put here initial = len(self.freqs)?
        self.ROI_s_z = self.settings.New('ROI_s_z', dtype=int, initial=200, unit = 'px' )
        self.ROI_s_y = self.settings.New('ROI_s_y', dtype=int, initial=600, unit = 'px' )
        self.transpose_pattern = self.settings.New('transpose_pattern', dtype=bool, initial=False )
        self.exposure = self.settings.New("exposure", dtype = float, initial=100, vmin=1.004, vmax = 1e4, spinbox_step=10, spinbox_decimals=3, unit="ms")
        self.add_operation("read_subarray_vsize", self.read_subarray_vsize)
        self.settings.New('edge_trigger_margin', dtype = float, initial = self.calculate_margin(), vmin = 0.0, ro=True, spinbox_decimals = 3 , unit = 'ms')
        self.settings.New('effective_fps', dtype = float, initial = self.calculate_eff_fps(), vmin = 0.0, ro = True, spinbox_decimals = 2, unit = 'fps')
        self.settings.New('skip_upload', dtype=bool, initial=False )
        self.time_lapse = self.settings.New('time_lapse', dtype = bool, initial = False)
        self.obs_time = self.settings.New('obs_time', dtype=float, initial= 0.0, vmin = 0, spinbox_decimals = 3, spinbox_step = 10.0,  unit = 's' )
        self.dark_time = self.settings.New('dark_time', dtype = float, initial = 0.0, vmin = 0, spinbox_decimals = 3, spinbox_step = 1, unit = 's')
        self.time_frames_n  = self.settings.New('time_frames_n', dtype = int, initial = 1, vmin = 1, ro = True)
 
        #set functions
        self.sheet_width.hardware_set_func = self.set_sheet_width
        self.PosNeg.hardware_set_func = self.set_PosNeg
        self.ROI_s_z.hardware_set_func = self.set_ROI_s_z
        self.ROI_s_y.hardware_set_func = self.set_ROI_s_y
        self.transpose_pattern.hardware_set_func = self.set_transpose_pattern
        self.exposure.hardware_set_func = self.set_exposure
        self.time_lapse.hardware_set_func = self.set_time_lapse
        self.obs_time.hardware_set_func = self.set_obs_time
        self.dark_time.hardware_set_func = self.set_dark_time
        
    
    def setup_figure(self):
        """
        Runs once during App initialization, after setup()
        This is the place to make all graphical interface initializations,
        build plots, etc.
        """
                
        # connect ui widgets to measurement/hardware settings or functions
        self.ui.start_pushButton.clicked.connect(self.start)
        self.ui.interrupt_pushButton.clicked.connect(self.interrupt)
        # self.settings.save_h5.connect_to_widget(self.ui.save_h5_checkBox)
        
        # connect ui widgets of live settings
        self.settings.auto_levels.connect_to_widget(self.ui.autoLevels_checkBox)
        self.settings.auto_range.connect_to_widget(self.ui.autoRange_checkBox)
        self.settings.level_min.connect_to_widget(self.ui.min_doubleSpinBox) #spinBox doesn't work nut it would be better
        self.settings.level_max.connect_to_widget(self.ui.max_doubleSpinBox) #spinBox doesn't work nut it would be better
        
        # Set up pyqtgraph graph_layout in the UI
        self.imv = pg.ImageView()
        self.ui.plot_groupBox.layout().addWidget(self.imv)
        
        
    def update_display(self):
        """
        Displays the numpy array called self.image.  
        This function runs repeatedly and automatically during the measurement run,
        its update frequency is defined by self.display_update_period.
        """
        
        self.display_update_period = self.settings['refresh_period'] 
        
        # length = self.camera.hamamatsu.number_image_buffers
        length = self.camera.number_frames.val
        if hasattr(self, 'frame_index'):
            self.settings['progress'] = (self.frame_index +1) * 100/length 
        
        if hasattr(self, 'image'):
            self.imv.setImage(self.image,
                                autoLevels = self.settings.auto_levels.val,
                                autoRange = self.settings.auto_range.val,
                                levelMode = 'mono'
                                )
            
            if self.settings['auto_levels']:
                lmin,lmax = self.imv.getHistogramWidget().getLevels()
                self.settings['level_min'] = lmin
                self.settings['level_max'] = lmax
            else:
                self.imv.setLevels( min= self.settings['level_min'],
                                    max= self.settings['level_max'])
            
             
    def run(self):
        
        
        print('\n\n======================================\n  DMD light sheet measurement begins\n======================================\n\n')
        
        self.frame_index = -1
        self.eff_subarrayh = int(self.camera.subarrayh.val/self.camera.binning.val)
        self.eff_subarrayv = int(self.camera.subarrayv.val/self.camera.binning.val)
        
        
        
        # get a single image to initialize the acquisition and the h5 file
        exposure = self.settings.exposure.val
        self.camera.settings['number_frames'] = 1
        self.camera.settings['exposure_time'] = exposure*1e-3   # This could be uselful to determine the dark counts?
        self.camera.read_from_hardware()
        self.camera.hamamatsu.startAcquisition()
        frame_index = 0
        [frame, dims] = self.camera.hamamatsu.getLastFrame()        
        self.np_data = frame.getData()
        self.image = np.reshape(self.np_data,(self.eff_subarrayv, self.eff_subarrayh))
        self.camera.hamamatsu.stopAcquisition()
        
        
        if not self.settings['transpose_pattern']:
             self.settings['num_frames'] = (1 + self.settings['PosNeg']) * int(np.floor(self.settings['ROI_s_z']/self.settings['sheet_width']))
        else:
             self.settings['num_frames'] = (1 + self.settings['PosNeg']) * int(np.floor(self.settings['ROI_s_y']/self.settings['sheet_width']))
             
        num_frames = self.settings['num_frames']
        
        self.settings['edge_trigger_margin'] = self.calculate_margin()
        self.dmd_hw.settings['exposure'] = int(exposure*1e3 + self.settings['edge_trigger_margin']*1e3)
        exposure_dmd = [int(exposure*1e3 + self.settings['edge_trigger_margin']*1e3) ]*num_frames
        self.settings['effective_fps'] =  self.calculate_eff_fps()
        
        # I tell the camera how manìy frames to record
        self.camera.settings['number_frames'] = num_frames
        self.camera.settings['exposure_time'] = exposure*1e-3
        
        
        # =============================================================================
        #        upload patterns 
        # =============================================================================
                
        
        if not self.settings['skip_upload'] or self.settings['time_lapse']:
        
            print("\n****************\nLoading pattern\n****************\n")
            t_load_init = time.time()
    
            dark_time = [self.dmd_hw.dark_time.val]*num_frames
            trigger_input = [self.dmd_hw.trigger_input.val]*num_frames
            trigger_output = [self.dmd_hw.trigger_output.val]*num_frames
            rep = num_frames
            
            transpose_pattern = self.settings['transpose_pattern']
    
            ROI_size = [self.settings['ROI_s_z'], self.settings['ROI_s_y']]
            
            print(f'\nCreating {num_frames} patterns...')
            print('\nPosNeg: ', self.settings['PosNeg'])
            print('Transpose: ', self.settings['transpose_pattern'])
            print('\nPlease wait...', end = '')
            
            
            
            if self.settings['PosNeg'] == False:
                 
                images = create_light_sheets(ROI_size , self.settings['sheet_width'], transpose_pattern)
                
            else:
                #PosNeg
                images = []
                im_pos = create_light_sheets(ROI_size , self.settings['sheet_width'], transpose_pattern)
                
                mask = create_rectangle_mask(ROI_size)
                
                for im in im_pos:
                    images.append(im)
                    im_neg = np.uint8(np.logical_not(im)*1)
                    images.append(im_neg*mask)
                        
            print(f'     >>     Pattern creation completed ({time.time() - t_load_init:.3f} s)\n')
            
    
            self.dmd_hw.dmd.def_sequence(images, exposure_dmd,trigger_input,dark_time,trigger_output,rep)
            
            print("\n****************\nLoad sheets ends\n****************\n")
            
        elif not self.settings['time_lapse'] :
            print('WARNING!\nNo images uploaded: the DMD uses the last uploaded sequence of patterns')
           
       
        self.initH5() 
           
        # for loop for time lapse
    
        
        t_init = time.time()  # we record the initial time after the first upload
        
        time_frames_n = self.settings['time_frames_n']
        for time_index in range(time_frames_n):
            
            print(f' --- Volume acqusition number {time_index+1}/{time_frames_n} ---')
            
            # Set trigger to external
    
            self.camera.hamamatsu.setTriggerSource("external")
            self.camera.hamamatsu.setTriggerMode("normal")
            self.camera.hamamatsu.setTriggerPolarity("positive")
            self.camera.hamamatsu.setTriggerActive("edge")
    
            print('\nTrigger set to external: edge mode')
            
            #=====================================
            # Start acquisition
            #=====================================
            
            print('\nAcquisition starts')
            
            self.frame_index = -1
            self.eff_subarrayh = int(self.camera.subarrayh.val/self.camera.binning.val)
            self.eff_subarrayv = int(self.camera.subarrayv.val/self.camera.binning.val)
            self.camera.read_from_hardware()
            self.camera.hamamatsu.startAcquisition()
        
            frame_index = 0
            
            #=====================================
            # Shutter open
            #=====================================
            
            print('\nShutter open')
            
            t_shutter_open = time.time()
            
            self.shutter_hw.shutter.open_shutter()
            time.sleep(0.3) #seconds
            
            #=====================================
            # DMD start
            #=====================================
            
            self.dmd_hw.dmd.startsequence()
            
            #=====================================
            # Get and save frames
            #=====================================
            
            print('\nSaving frames')
            
            while frame_index < self.camera.hamamatsu.number_image_buffers:
    
                # Get frames.
                #The camera stops acquiring once the buffer is terminated (in snapshot mode)
                [frames, dims] = self.camera.hamamatsu.getFrames()
                
               
                for aframe in frames:
                    
                    self.np_data = aframe.getData()  
                    self.image = np.reshape(self.np_data,(self.eff_subarrayv, self.eff_subarrayh)) 
                    
                    self.image_h5[time_index][frame_index,:,:] = self.image
                    self.h5file.flush() 
                                        
                    frame_index += 1
                    self.frame_index = frame_index 
                    
                    print(frame_index)
                
                    if self.interrupt_measurement_called:
                        break    
       
            #=====================================
            # Shutter close
            #=====================================
            
            
            self.shutter_hw.shutter.close_shutter()
            print(f'\nShutter closed. Opened for {time.time() - t_shutter_open:.3f} s')
            
            #=====================================
            # Stop acquisition
            #=====================================
            
            self.camera.hamamatsu.stopAcquisition()
    
            #=====================================
            # Set trigger to internal
            #=====================================
            
            
            self.camera.hamamatsu.setTriggerSource("internal")
            self.camera.hamamatsu.setTriggerMode("normal")
            self.camera.hamamatsu.setTriggerPolarity("positive")
            self.camera.hamamatsu.setTriggerActive("edge")
            print('\nTrigger set to internal')
            
            
            
            #=====================================
            # exit conditions
            #=====================================
            
            if not self.settings['time_lapse']:
                break # redundant since if setting time_lapse is false time_framse_n = 1
                
            else:
                time.sleep(self.settings['dark_time']) #seconds
                time_end_of_frame = time.time() - t_init
                print(f'Total time elapsed: {time_end_of_frame:.3f} s\n')
          
        #out of for loop
        self.h5_group.attrs['measure_duration'] = time_end_of_frame
        self.h5file.close()     
        
        
        
        print('\n========================================\nDMD light sheet measurement has finished\n========================================\n\n')
        
        
        
        
        
        
        
        
        
        
        
        
    
    def initH5(self):
        
        def create_saving_directory():
            if not os.path.isdir(self.app.settings['save_dir']):
                os.makedirs(self.app.settings['save_dir'])    
        
        create_saving_directory()
        
        # file name creation
        timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
        sample = self.app.settings['sample']
        if sample == '':
            sample_name = '_'.join([timestamp, self.name])
        else:
            sample_name = '_'.join([timestamp, self.name, sample])
        fname = os.path.join(self.app.settings['save_dir'], sample_name + '.h5')
        
        # file creation
        self.h5file = h5_io.h5_base_file(app=self.app, measurement=self, fname = fname)
        self.h5_group = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5file)
        
        img_size = self.image.shape
        length = self.settings['num_frames']

        self.image_h5 = [None]*self.settings['time_frames_n']
        
        for time_index in range(self.settings['time_frames_n']):
            
            name = f't{time_index:04d}/c0000/image'
            self.image_h5[time_index] = self.h5_group.create_dataset( name  = name, 
                                                          shape = ( length, img_size[0], img_size[1]),
                                                          dtype = self.image.dtype, chunks = (1, self.eff_subarrayv, self.eff_subarrayh)
                                                          )
            self.image_h5[time_index].dims[0].label = "z"
            self.image_h5[time_index].dims[1].label = "y"
            self.image_h5[time_index].dims[2].label = "x"
            self.image_h5[time_index].attrs['element_size_um'] =  [1,1,1] # required for compatibility with imageJ
        
        