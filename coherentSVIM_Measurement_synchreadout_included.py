""" 
   Written by Michele Castriotta, Alessandro Zecchi, Andrea Bassi (Polimi).
   Code for creating the measurement class of ScopeFoundry for the Orca Flash 4V3
   11/18
"""

from ScopeFoundry import Measurement
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file
from ScopeFoundry import h5_io
import pyqtgraph as pg
import numpy as np
import os
import time
        



def create_squared_pattern_from_freq(num_of_periods = 1, transpose_pattern=False, cropped_field_size = [270, 810],
                           centered = True, phase = 0.0, odd = False, im_size = [1080, 1920]):
    """    
    We have to generate a square wave of given period with inclination of 45° over
    a frame of dimensions 1080x1920
    
    Each square pixel has border length s_pixel = 7.56 um
    The unit period is period_unit = sqrt(2)*s_pixel = 10.7 um
    
    OUTPUTS:
        - image             : 8bit image (numpy.uint8 array) with the desired pattern
        - displayed_per_num : (float) Actual number of periods displayed in the ROI rectangle
    
    PARAMETERS:
        - num_of_periods    : number of periods in the cropped ROI width
        - transpose_pattern : FASLE -> antiDIAG, TRUE -> DIAG
        - centered          : TRUE -> set the origin of the square wave to the center of the
                              cropped rectangle. FALSE -> the origin will be the upper left
                              corner of the cropped rectangle
        - phase             : offset phase expressed as a fraction of the period (0.5 -> 180deg)
        - odd               : If True the pattern is odd with respect to the 45deg line going
                              throught the center of the image
        - cropped_field_size: dimentions of the rectangle ROI. First dimention refers to the border
                              parallel to the diagonal direction and the second to the border parallel to the
                              anti-diagonal direction. Default is [270,810] to have the largest rectangle with
                              borders proportioned 3:1
        - im_size           : Size of the outpun image in pizel; default is [1080,1920]
        
    """
    
    s_y = im_size[0]
    s_x = im_size[1]
    
    # dimentions of the rectangle to be cropped out in units of pizel diagonal (same as unit_period)
    s_diag = cropped_field_size[0] #dimension of the border parallel to the diagonal direction
    s_anti = cropped_field_size[1] #dimension of the border parallel to the antidiagonal direction  


    
    
    # uniform illumination
    if num_of_periods == 0:
        return np.ones(im_size, dtype = 'uint8'), 0
    


    image = np.zeros(im_size, dtype = 'uint8')
    # image = np.zeros(size, dtype = 'bool')
    
    if not transpose_pattern and num_of_periods:
        
        # antidiag
        
        # t = time.time()
        
        period = int(s_diag/num_of_periods)
        disp_per_num = s_diag/period
        
        shift = ( s_y + ( s_x - s_y )/2  - (not odd and centered)*(int(period/2))) - (not centered)*s_diag + int(2*period*phase)
        
        # I create the single strip to shift
        
        x_coord = np.linspace(-shift, (s_x + s_y - shift -1), s_x + s_y)
        strip = np.uint8(( x_coord%(2*period) < period)*1)
        # strip = ( x_coord%(2*period) < period)
        
        # I move the squarewave strip to create the 45deg angle
         
        for i in range(s_y):
            image[i, :] = strip[(s_y-i-1):(s_y + s_x -i-1)]
             
            
        # print(f'Time for creation of uncropped image: {time.time() - t}')
    
    else:    
        
        # diag
        
        # t = time.time()
        
        period = int(s_anti/num_of_periods)
        disp_per_num = s_anti/period
        
        shift = ( (s_y + s_x)/2  - (not odd and centered)*(int(period/2)) ) - (not centered)*s_anti  + int(2*period*phase)
        
        x_coord = np.linspace(-shift, (s_x + s_y - shift -1), s_x + s_y)
        strip = np.uint8(( x_coord%(2*period) < period)*1)
        # strip = ( x_coord%(2*period) < period)
        
        for i in range(s_y):
            image[i, :] = strip[(i):( s_x +i)]
            
            
        # print(f'Time for creation of uncropped image: {time.time() - t}')    
                        
    return image, disp_per_num



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
    
    



class coherentSvimMeasurement(Measurement):     
    
    name = "coherent_SVIM"
    
    
    def set_f_min(self,f_min):
        # self.settings['f_min'] = f_min
        
        if hasattr(self, 'num_frames'):
            self.settings['num_frames']  = self.settings['f_max'] - f_min + 1
            
    def set_f_max(self,f_max):
        # self.settings['f_max'] = f_max
        
        if hasattr(self, 'num_frames'):
           self.settings['num_frames']  = f_max - self.settings['f_min'] + 1
     
           
    def calculate_margin(self):
        delay =  87.7 #(us)
        read_one_line = 9.74436 #(us)
        contingency = 0.005
        self.eff_subarrayv = int(self.camera.subarrayv.val/self.camera.binning.val)
        
        return (1 + contingency) * (delay + (self.eff_subarrayv/2) * read_one_line) *1e-3 #(ms)


    
    def set_exposure(self, exposure):

        
        self.settings['edge_trigger_margin'] = self.calculate_margin()
        
        if self.settings['trigger_active'] == 'edge':
                self.settings['effective_fps'] =  1000/(exposure + self.settings['edge_trigger_margin'])
        elif self.settings['trigger_active'] == 'syncreadout':
                self.settings['effective_fps'] =  1000/(exposure)
                
    def set_trigger_active(self, trigger_active)  :
        
        if self.settings['trigger_active'] == 'edge':
                self.settings['effective_fps'] =  1000/(self.settings['exposure'] + self.settings['edge_trigger_margin'])
        elif self.settings['trigger_active'] == 'syncreadout':
                self.settings['effective_fps'] =  1000/(self.settings['exposure'])
        
        
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
        self.f_min = self.settings.New('f_min', dtype=int, initial=0 )
        self.f_max = self.settings.New('f_max', dtype=int, initial=10 )
        self.num_frames = self.settings.New('num_frames',ro = True, dtype = int, initial = 11) # TODO make autoupdate
        self.settings.New('ROI_s_z', dtype=int, initial=200, unit = 'px' )
        self.settings.New('ROI_s_y', dtype=int, initial=600, unit = 'px' )
        self.settings.New('transpose_pattern', dtype=bool, initial=True )  # TODO revert to initial False
        self.exposure = self.settings.New("exposure", dtype = float, initial=100, vmin=1.004, vmax = 1e4, spinbox_step=10, spinbox_decimals=3, unit="ms")
        self.trigger_active = self.settings.New('trigger_active', dtype=str, si=False, ro=0, 
                          choices = ["edge", "syncreadout"], initial = 'edge', reread_from_hardware_after_write = True)
        self.settings.New('edge_trigger_margin', dtype = float, initial = self.calculate_margin(), vmin = 0.0, ro=True, spinbox_decimals = 3 , unit = 'ms')
        self.effective_fps = self.settings.New('effective_fps', dtype = float, initial = 0.0, vmin = 0.0, ro = True, spinbox_decimals = 1, unit = 'fps') # TODO make autoupdate
        
 
        #set functions
        self.f_min.hardware_set_func = self.set_f_min
        self.f_max.hardware_set_func = self.set_f_max
        self.exposure.hardware_set_func = self.set_exposure
        self.trigger_active.set_func = self.set_trigger_active
        
        
        
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
        
        
        print('\n\n======================================\n   Coherent SVIM measurement begins\n======================================\n\n')
        
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
        
        print("\n****************\nLoading pattern\n****************\n")
        t = time.time()
        f_start = self.settings['f_min']
        f_stop = self.settings['f_max']
        freqs = np.linspace(f_start, f_stop, f_stop - f_start + 1,dtype = int)
        phases = np.linspace(0,1, self.dmd_hw.number_of_phases.val , endpoint = False, dtype = float) # they will be used as fractions of the period  
        num_frames = len(freqs) * self.dmd_hw.number_of_phases.val
        
        
        
        if self.settings['trigger_active'] == 'edge':
            num_frames_dmd = num_frames
            self.dmd_hw.settings['exposure'] = int(exposure*1e3 + self.settings['edge_trigger_margin']*1e3) # TODO: check this extra time
            exposure_dmd = [int(exposure*1e3 + self.settings['edge_trigger_margin']*1e3) ]*num_frames_dmd
        
        elif self.settings['trigger_active'] == 'syncreadout':
            num_frames_dmd = num_frames + 1 # I add one frame for the to terminate the last exposure with the trigger from a black frame
            # DMD exposure and camera exposure are the same
            self.dmd_hw.settings['exposure'] = int(exposure*1e3)
            exposure_dmd = [int(exposure*1e3) ]*num_frames_dmd
        
        dark_time = [self.dmd_hw.dark_time.val]*num_frames_dmd
        trigger_input = [self.dmd_hw.trigger_input.val]*num_frames_dmd
        trigger_output = [self.dmd_hw.trigger_output.val]*num_frames_dmd
        
        rep = num_frames_dmd
        self.dmd_hw.settings['sequence_repetitions'] = 1 # Update the HW setting  
        
       
        # I tell the camera how manìy frames to record
        self.camera.settings['number_frames'] = num_frames
        self.camera.settings['exposure_time'] = exposure*1e-3
        
        
        
        transpose_pattern = self.settings['transpose_pattern']
        self.dmd_hw.settings['transpose_pattern'] = transpose_pattern
        self.dmd_hw.settings['crop_squared'] = True # I always crop

        crop_squared_size = [self.settings['ROI_s_z'], self.settings['ROI_s_y']]
          
        if self.dmd_hw.squared_pattern_origin.val == 'rectangle_center':
            centered = True
        else:
            centered = False
        
        
        
        print(f'\nCreating {num_frames} patterns...\n\nTheoretical frequencies: ', freqs)
        print('\nPhases: ', phases)
        print('\nPlease wait...', end = '')
        
        mask = create_rectangle_mask(crop_squared_size)
        
        images = []
        freqs_out = []
        
        if self.dmd_hw.squared_scan_mode.val == 'periods_then_phases':
             for phase in phases:
                # print('\n')
                for freq in freqs:
                    # print(f'Creating square wave with freq   {freq:3d}   and phase shift   {phase:.2f}')
                    
                    im, freq_out = create_squared_pattern_from_freq(freq,transpose_pattern, crop_squared_size, centered, phase)
                    
                    images.append(im)
                    freqs_out.append(freq_out)
        
        elif self.dmd_hw.squared_scan_mode.val == 'phases_then_periods':
            for freq in freqs:
                # print('\n')
                for phase in phases:
                    # print(f'Creating square wave with freq   {freq:3d}   and phase shift   {phase:.2f}')
                    
                    im, freq_out = create_squared_pattern_from_freq(freq,transpose_pattern, crop_squared_size, centered, phase)
                    
                    images.append(im)
                    freqs_out.append(freq_out)
                    
                    
        images_arr = np.array(images)
        cropped_images = list(images_arr*mask)
            
        print(f'     >>     Pattern creation completed ({time.time() - t:.3f} s)\n')
        print('The actual displayed frequencies are: ', freqs_out)
        
        
        
        
        if self.settings['trigger_active'] == 'syncreadout':
            # additional black frame to give the final trigger that stops the exposure of the last frame
            cropped_images.append(np.zeros([1080, 1920], dtype = 'uint8'))
        
        
        
        self.dmd_hw.dmd.def_sequence(cropped_images, exposure_dmd,trigger_input,dark_time,trigger_output,rep)
        
        print("\n****************\nLoad squared ends\n****************\n")
        
        
        
        # create_squared_pattern_from_freq(num_of_periods = 1, transpose_pattern=False, cropped_field_size = [270, 810],
                           # centered = True, phase = 0.0, odd = False, im_size = [1080, 1920]):
        
                               
        
        
        
        #=====================================
        # Set trigger to external
        #=====================================
        
        
        if self.settings['trigger_active'] == 'edge':
            
            self.camera.hamamatsu.setTriggerSource("external")
            self.camera.hamamatsu.setTriggerMode("normal")
            self.camera.hamamatsu.setTriggerPolarity("positive")
            self.camera.hamamatsu.setTriggerActive("edge")
        
        elif self.settings['trigger_active'] == 'syncreadout':
            
            # For sych read out we can achieve frame rate of circa (11ms)^-1 (circa 100fps)
            # We add a black frame at the end of the DMD sequence to terminate the 
            # exposure of the last camera frame
            
            self.camera.hamamatsu.setTriggerSource("external")
            self.camera.hamamatsu.setTriggerMode("normal")
            self.camera.hamamatsu.setTriggerPolarity("positive")
            self.camera.hamamatsu.setTriggerActive("syncreadout")  
        
        
        print('\nTrigger set to external: ', self.settings['trigger_active'] )
        
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
        
        self.initH5()

        #=====================================
        # Shutter open
        #=====================================
        
        print('\nShutter open')
        
        t_shutter_open = time.time()
        
        self.shutter_hw.shutter.open_shutter()
        
        
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
                
                self.image_h5[frame_index,:,:] = self.image
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
        self.h5file.close() 

        #=====================================
        # Set trigger to internal
        #=====================================
        
        
        self.camera.hamamatsu.setTriggerSource("internal")
        self.camera.hamamatsu.setTriggerMode("normal")
        self.camera.hamamatsu.setTriggerPolarity("positive")
        self.camera.hamamatsu.setTriggerActive("edge")
        print('\nTrigger set to internal')
        
        
        
        #=====================================
        # Stop DMD pattern ??????
        #=====================================
        
        # self.dmd_hw.dmd.stopsequence()
        
        
        
        print('\n======================================\nCoherent SVIM measurement has finished\n======================================\n\n')
        
        
        
        
        
        
        
        
        
        
        
        
    
    def initH5(self):
        """
        Initialization operations for the h5 file.
        """
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
        length = self.camera.hamamatsu.number_image_buffers
        self.image_h5 = self.h5_group.create_dataset( name  = 't0/c0/image', 
                                                      shape = ( length, img_size[0], img_size[1]),
                                                      dtype = self.image.dtype, chunks = (1, self.eff_subarrayv, self.eff_subarrayh)
                                                      )
        
        self.image_h5.dims[0].label = "z"
        self.image_h5.dims[1].label = "y"
        self.image_h5.dims[2].label = "x"
        
        #self.image_h5.attrs['element_size_um'] =  [self.settings['zsampling'], self.settings['ysampling'], self.settings['xsampling']]
        self.image_h5.attrs['element_size_um'] =  [1,1,1] # required for compatibility with imageJ