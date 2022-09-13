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
    
    



class BaseSvimMeasurement(Measurement):     
    
    name = "SVIM"
    
    def calculate_num_frames(self):
        return 64
    
    
    def calculate_time_frames_n(self):
        
        if not self.settings['time_lapse']:
            return int(1)
        else:
            delay = (not self.settings['keep_shutter_open']) * 0.3 +  0.3  # 0.3s is the trigger dead time we set, 0.3 is an empirical dead computational time
            return int(np.ceil( self.settings['obs_time'] / ( (self.settings['num_frames']/self.settings['effective_fps']) + self.settings['dark_time'] + delay ) ))  
     
    
                
    def set_PosNeg(self, PosNeg):
        
        if hasattr(self, 'num_frames') and hasattr(self, 'transpose_pattern'):
            self.settings['num_frames'] = self.calculate_num_frames()
            
        if hasattr(self, 'time_frames_n'):
            self.settings['time_frames_n'] = self.calculate_time_frames_n()
                
    def set_ROI_s_z(self, ROI_s_z):
        
        if hasattr(self, 'transpose_pattern'):
            self.settings['num_frames'] = self.calculate_num_frames()
        if hasattr(self, 'time_frames_n'):
            self.settings['time_frames_n'] = self.calculate_time_frames_n()
            
    def set_ROI_s_y(self, ROI_s_y):
        
        if hasattr(self, 'transpose_pattern'):
            self.settings['num_frames'] = self.calculate_num_frames()
        if hasattr(self, 'time_frames_n'):
            self.settings['time_frames_n'] = self.calculate_time_frames_n()
            
    def set_transpose_pattern(self, transpose):
        
        self.settings['num_frames'] = self.calculate_num_frames()
              
        if hasattr(self, 'time_frames_n'):
                self.settings['time_frames_n'] = self.calculate_time_frames_n()
                
    def set_comp_sensing(self, comp_sensing):
        
        self.settings['num_frames'] = self.calculate_num_frames()
        
    def set_cs_subset_dim(self, cs_subset_dim):
        # this function should be overwritten in the specific measurement to set
        # the specific maximum value (e.g. the dimention of the hadamard basis)
        
        self.settings['num_frames'] = self.calculate_num_frames()
           
    def calculate_margin(self):
        
        read_one_line = 9.74436 #(us)
        delay =  9 * read_one_line #(us)
        contingency = 0.04
        
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
    
    def set_keep_shutter_open(self, keep_shutter_open):
        
        if hasattr(self, 'time_frames_n'):
            self.settings['time_frames_n'] = self.calculate_time_frames_n()
            
            
     
        
    def setup(self):
        
        "..."
        self.ui_filename = sibling_path(__file__, "coherentSVIM.ui")
        self.camera = self.app.hardware['HamamatsuHardware']
        self.dmd_hw = self.app.hardware['TexasInstrumentsDmdHW']
        self.shutter_hw = self.app.hardware['Shutter']
        self.ui = load_qt_ui_file(self.ui_filename)
        self.setup_svim_mode_settings()
        self.PosNeg = self.settings.New('PosNeg', dtype = bool, initial = True)
        self.ROI_s_z = self.settings.New('ROI_s_z', dtype=int, initial=64, unit = 'px' )
        self.settings.New('ROI_s_y', dtype=int, initial=600, unit = 'px' )
        self.transpose_pattern = self.settings.New('transpose_pattern', dtype=bool, initial=False )
        self.comp_sensing = self.settings.New('comp_sensing', dtype = bool, initial = False )
        self.cs_subset_dim = self.settings.New('cs_subset_dim', dtype = int, initial = 4, vmin = 1)
        self.num_frames = self.settings.New('num_frames',ro = True, dtype = int, initial = self.calculate_num_frames())    # TODO The initial value of this setting is critical: so far it must be updated manually if one changes any other initial value. Should we calculate self.freqs during the setup period and put here initial = len(self.freqs)?
        self.exposure = self.settings.New("exposure", dtype = float, initial=100, vmin=1.004, vmax = 1e4, spinbox_step=10, spinbox_decimals=3, unit="ms")
        self.add_operation("read_subarray_vsize", self.read_subarray_vsize)
        self.settings.New('edge_trigger_margin', dtype = float, initial = self.calculate_margin(), vmin = 0.0, ro=True, spinbox_decimals = 3 , unit = 'ms')
        self.settings.New('effective_fps', dtype = float, initial = self.calculate_eff_fps(), vmin = 0.0, ro = True, spinbox_decimals = 2, unit = 'fps')
        self.settings.New('skip_upload', dtype=bool, initial=False )
        self.time_lapse = self.settings.New('time_lapse', dtype = bool, initial = False)
        self.obs_time = self.settings.New('obs_time', dtype=float, initial= 5.0, vmin = 0, spinbox_decimals = 3, spinbox_step = 10.0,  unit = 's' )
        self.dark_time = self.settings.New('dark_time', dtype = float, initial = 0.0, vmin = 0, spinbox_decimals = 3, spinbox_step = 1, unit = 's')
        self.time_frames_n  = self.settings.New('time_frames_n', dtype = int, initial = 1, vmin = 1, ro = True)
        self.keep_shutter_open = self.settings.New('keep_shutter_open', dtype = bool, initial = True)
        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals=4, initial=0.04, vmin=0)
        self.settings.New('auto_range', dtype=bool, initial=True )
        self.settings.New('auto_levels', dtype=bool, initial=True )
        self.settings.New('level_min', dtype=int, initial=60 )
        self.settings.New('level_max', dtype=int, initial=150 )
        #set functions
        self.PosNeg.hardware_set_func = self.set_PosNeg
        self.exposure.hardware_set_func = self.set_exposure
        self.ROI_s_z.hardware_set_func = self.set_ROI_s_z
        self.transpose_pattern.hardware_set_func = self.set_transpose_pattern
        self.comp_sensing.hardware_set_func = self.set_comp_sensing
        self.cs_subset_dim.hardware_set_func = self.set_cs_subset_dim
        self.time_lapse.hardware_set_func = self.set_time_lapse
        self.obs_time.hardware_set_func = self.set_obs_time
        self.dark_time.hardware_set_func = self.set_dark_time
        self.keep_shutter_open.hardware_set_func = self.set_keep_shutter_open
        # TODO This does not actually work, the wrong initial value for num_frames is not corrected! 
        self.settings['PosNeg'] = True # TODO decide if this is a good solution for the problem of self.num_frames updating on setup. This line should also ensures that self.freqs is created 
    
        
    def setup_svim_mode_settings(self):
        pass
    
    
    
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
        
        
        # this is the number of frames that will be taken by the camera (e.g. the dimention of the subset for CS)
        num_frames = self.settings['num_frames']
        
        
        # the DMD here needs the dimention of the complete basis that we want to upload
        if hasattr(self, 'load_num_frames'):
            dmd_num_frames = self.load_num_frames
        else:
            # this is in case CS is not implemented for a type of SVIM measurement
            dmd_num_frames = num_frames
        
        self.settings['edge_trigger_margin'] = self.calculate_margin()
        self.dmd_hw.settings['exposure'] = int(exposure*1e3 + self.settings['edge_trigger_margin']*1e3)
        exposure_dmd = [int(exposure*1e3 + self.settings['edge_trigger_margin']*1e3) ]*dmd_num_frames
        self.settings['effective_fps'] =  self.calculate_eff_fps()
        
        # I tell the camera how many frames to record
        self.camera.settings['number_frames'] = num_frames
        self.camera.settings['exposure_time'] = exposure*1e-3
        
        # =============================================================================
        #        upload patterns 
        # =============================================================================
        
        
        if not self.settings['skip_upload'] or self.settings['time_lapse']:
        
            print("\n****************\nLoading pattern\n****************\n")
            t_load_init = time.time()
            
            dark_time = [self.dmd_hw.dark_time.val]*dmd_num_frames
            trigger_input = [self.dmd_hw.trigger_input.val]*dmd_num_frames
            trigger_output = [self.dmd_hw.trigger_output.val]*dmd_num_frames
            rep = dmd_num_frames
           
            transpose_pattern = self.settings['transpose_pattern']
    
            crop_squared_size = [self.settings['ROI_s_z'], self.settings['ROI_s_y']]
              
            if self.dmd_hw.squared_pattern_origin.val == 'rectangle_center':
                centered = True
            else:
                centered = False
            
            print('\nPosNeg: ', self.settings['PosNeg'])
            print('Transpose: ', self.settings['transpose_pattern'])
            print('\nPlease wait...', end = '')
            
            mask = create_rectangle_mask(crop_squared_size)

            images = self.run_svim_mode_function()
                        
            images_arr = np.array(images)
            cropped_images = list(images_arr*mask)
                
            print(f'     >>     Pattern creation completed ({time.time() - t_load_init:.3f} s)\n')
            
    
            self.dmd_hw.dmd.def_sequence(cropped_images, exposure_dmd,trigger_input,dark_time,trigger_output,rep)
            
            print("\n****************\nLoad squared ends\n****************\n")
            
        elif not self.settings['time_lapse'] :
            print('WARNING!\nNo images uploaded: the DMD uses the last uploaded sequence of patterns')

        
        self.initH5()
        
        if self.settings['time_lapse'] == True:
            print('Keep shutter open: ',self.settings['keep_shutter_open'])

        # while loop for time laps
        time_index = -1

        t_init = time.time()  # we record the initial time after the first upload
        while True:
            
            time_index += 1 
            print(f' --- Volume acqusition number {time_index + 1} ---')
            
            
            # Compressed sensing choice of basis subset
            
            if self.settings['comp_sensing'] == True:
            
                # "sequence" tells how to reorder and/or choose a subset of the loaded patterns for the current time frame
                # NB: the pattern number starts from 0
                
                # sequence = [3,2,1,0] 
                
                sequence = self.run_iteration_cs_sequence(time_index)
                print(f'\nPattern sequence for time point {time_index+1}: ', end = '')
                print(sequence)
                
                if time_index == 0:
                    self.CS_time_point_subsets = []
                    
                self.CS_time_point_subsets.append(sequence.copy())
                
                if self.settings['PosNeg'] == True:
                    
                    temp = []
                    for i in sequence:
                        temp.append(i*2)
                        temp.append(i*2+1)
                        
                    sequence = temp
                
                repeatnum = len(sequence)
                # print(sequence)
                
                
                self.dmd_hw.dmd.reorderlut(sequence, repeatnum)
                
                
            
            
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
            
            if not self.settings['keep_shutter_open'] or time_index == 0:
            
                print('\nShutter open')
                t_shutter_open = time.time()
                self.shutter_hw.shutter.open_shutter()
                time.sleep(0.3) #seconds, dead time to open the shutter
            
            #=====================================
            # DMD start
            #=====================================
            
            self.dmd_hw.dmd.startsequence()
            
            #=====================================
            # Get and save frames
            #=====================================
            
            print('\nSaving frames')
            
            self.save_h5dataset(time_index)
                    
            while frame_index < self.camera.hamamatsu.number_image_buffers:
    
                # Get frames.
                #The camera stops acquiring once the buffer is terminated (in snapshot mode)
                [frames, dims] = self.camera.hamamatsu.getFrames()
                
               
                for aframe in frames:
                    
                    self.np_data = aframe.getData()  
                    self.image = np.reshape(self.np_data,(self.eff_subarrayv, self.eff_subarrayh)) 
                    self.h5file.flush() 
                    
                    self.image_h5[frame_index,:,:] = self.image
                        
                    frame_index += 1
                    self.frame_index = frame_index 
                    
                    print(frame_index)
                
                    if self.interrupt_measurement_called:
                        self.camera.hamamatsu.stopAcquisition()
                        self.camera.hamamatsu.setTriggerSource("internal")
                        self.camera.hamamatsu.setTriggerMode("normal")
                        self.camera.hamamatsu.setTriggerPolarity("positive")
                        self.camera.hamamatsu.setTriggerActive("edge")
                        print('\nTrigger set to internal')
                        self.shutter_hw.shutter.close_shutter()
                        break  
                    
                if self.interrupt_measurement_called:
                    break 
            
            
            #=====================================
            # Shutter close
            #=====================================
            
            if not self.settings['keep_shutter_open']:
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
                time_end_of_frame = time.time() - t_init
                print(f'Total time elapsed: {time_end_of_frame:.3f} s\n')
                break
                
            else:
                time.sleep(self.settings['dark_time']) #seconds
                time_end_of_frame = time.time() - t_init
                print(f'Total time elapsed: {time_end_of_frame:.3f} s\n')
                if time_end_of_frame > self.settings['obs_time']: #seconds
                    self.shutter_hw.shutter.close_shutter()
                    print(f'\nShutter closed. Opened for {time.time() - t_shutter_open:.3f} s')
                    break
              
            if self.interrupt_measurement_called:
                self.shutter_hw.shutter.close_shutter()
                print('\n\n>> MEASUREMENT INTERRUPTED <<\n\n')
                break
                
                
        # out of for loop
        self.h5_group.attrs['real_time_frames_n'] = time_index + 1
        
        if self.settings['comp_sensing'] == True:
            # print(self.CS_time_point_subsets)
            self.h5_group.attrs['CS_time_point_subsets'] = self.CS_time_point_subsets
        
        self.h5file.close()     
        
        
        
        print('\n======================================\nCoherent SVIM measurement has finished\n======================================\n\n')
        
 
    def run_svim_mode_function(self):
        pass
 
    def run_iteration_cs_sequence(self, **args):
        pass
 
    
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
        
    def save_h5dataset(self,t_index):
        
        img_size = self.image.shape
        length = self.settings['num_frames']
        
        name = f't{t_index:04d}/c0000/image'
        image_h5 = self.h5_group.create_dataset( name  = name, 
                                                      shape = ( length, img_size[0], img_size[1]),
                                                      dtype = self.image.dtype, chunks = (1, self.eff_subarrayv, self.eff_subarrayh)
                                                      )
        image_h5.dims[0].label = "z"
        image_h5.dims[1].label = "y"
        image_h5.dims[2].label = "x"
        image_h5.attrs['element_size_um'] =  [1,1,1] # required for compatibility with imageJ
        timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
        image_h5.attrs['timestamp'] = timestamp
        self.image_h5 = image_h5            