#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:23:38 2022

@author: marcovitali
"""

import sys
from qtpy import QtWidgets, uic
import qtpy.QtCore
import pyqtgraph as pg
from get_h5_data import get_h5_dataset, get_h5_attr
import h5py
from analyser_transform_6090 import coherentSVIM_analysis
from analyser_DMD_light_sheet import DMD_light_sheet_analysis

import numpy as np

from show_image import show_image

# import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas



class basic_app(coherentSVIM_analysis):
    
    def __init__(self, argv = []):
        
        self.qtapp = QtWidgets.QApplication(argv)
        self.name = 'basic_app'
        self.qtapp.setApplicationName(self.name)
        self.dialogs = list()
      
        
    def setup(self):
        
        self.ui_filename = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/smSVIM_microscope_analyser/analyser_6090_tabs.ui'
        self.ui = uic.loadUi(self.ui_filename)
        
        # file path and load
        
        self.file_path = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data'
        self.ui.pushButton_file_browser.clicked.connect(self.file_browser)
        self.ui.pushButton_load_dataset.clicked.connect(self.load_file_path)
        
        
# =============================================================================
#         setup: COHERENT SVIM
# =============================================================================
        
        
        self.params = {}
        
        # base selection
        
        self.ui.tabs.setCurrentWidget(self.ui.tab_coherent)
        
        self.bases = ['cos', 'sq', 'hadam']
        
        # select ROI
        
        def enable_ROI():
            self.select_ROI = self.ui.checkBox_select_ROI.isChecked()
            if self.select_ROI:
                self.ui.widget_ROI_params.setEnabled(True)
            else:
                self.ui.widget_ROI_params.setEnabled(False)
        self.select_ROI = False
        self.ui.checkBox_select_ROI.stateChanged.connect(enable_ROI)
        
        
        
        # denoise params
        self.ui.widget_denoise_params.setEnabled(False) # I'm not sure why this is needed
        def enable_denoise():
            self.denoise = self.ui.checkBox_denoise.isChecked()
            # print(self.denoise)
            if self.denoise:
                self.ui.widget_denoise_params.setEnabled(True)
            else:
                self.ui.widget_denoise_params.setEnabled(False)
        self.denoise = False
        self.ui.checkBox_denoise.stateChanged.connect(enable_denoise)
        
        
        # Invert single volume
        self.t_frame_index = 0
        def update_t_frame_index():  self.t_frame_index = self.ui.spinBox_t_frame_index.value()
        self.ui.spinBox_t_frame_index.valueChanged.connect(update_t_frame_index)
        
        self.ui.pushButton_show_raw_im.clicked.connect(self.show_im_raw_app)
        self.ui.pushButton_invert.clicked.connect(self.invert_volume_app)
        self.plot_modes = ['max','ave', 'stack']
        self.ui.pushButton_save_inverted.clicked.connect(self.save_inverted_app)
        self.ui.pushButton_show_inverted.clicked.connect(self.show_projections_app)
        
        
        
        # time lapse section
        self.time_lapse_modes = ['max','ave', 'plane']
        def change_tl_mode():
            if self.ui.comboBox_time_lapse_mode.currentIndex() != 2:
                self.ui.label_tl_plane.setEnabled(False)
                self.ui.spinBox_time_laps_plane.setEnabled(False)
            else:
                self.ui.label_tl_plane.setEnabled(True)
                self.ui.spinBox_time_laps_plane.setEnabled(True)
        
        self.ui.comboBox_time_lapse_mode.activated.connect(change_tl_mode)
        
        self.ui.pushButton_invert_time_lapse.clicked.connect(self.invert_tl_app)
        self.ui.pushButton_save_inverted_time_lapse.clicked.connect(self.save_time_lapse_app)
        self.ui.pushButton_show_time_lapse.clicked.connect(self.show_time_lapse)
        
        self.ui.label_status.setText('Please load dataset')
        
        
# =============================================================================
#         setup: DMD LIGHT SHEET
# =============================================================================
        
        
        def enable_ROI_ls():
            self.select_ROI_ls = self.ui.checkBox_select_ROI_ls.isChecked()
            if self.select_ROI_ls:
                self.ui.widget_ROI_params_ls.setEnabled(True)
            else:
                self.ui.widget_ROI_params_ls.setEnabled(False)
        self.select_ROI_ls = False
        self.ui.checkBox_select_ROI_ls.stateChanged.connect(enable_ROI_ls)
        
        self.t_frame_index_ls = 0
        def update_t_frame_index_ls():  self.t_frame_index_ls = self.ui.spinBox_t_frame_index_ls.value()
        self.ui.spinBox_t_frame_index_ls.valueChanged.connect(update_t_frame_index_ls)
        
        
        self.ui.pushButton_denoise_ls.clicked.connect(self.denoise_ls_app)
        self.ui.pushButton_save_volume_ls.clicked.connect(self.save_volume_ls_app)
        self.ui.pushButton_show_ls.clicked.connect(self.show_projections_ls_app)
        
        self.time_lapse_modes = ['max', 'ave', 'plane']
        def change_tl_mode_ls():
            if self.ui.comboBox_time_lapse_mode_ls.currentIndex() != 2:
                self.ui.label_tl_plane_ls.setEnabled(False)
                self.ui.spinBox_time_laps_plane_ls.setEnabled(False)
            else:
                self.ui.label_tl_plane_ls.setEnabled(True)
                self.ui.spinBox_time_laps_plane_ls.setEnabled(True)
        self.ui.comboBox_time_lapse_mode_ls.activated.connect(change_tl_mode_ls)       
                
        self.ui.pushButton_buils_time_lapse_ls.clicked.connect(self.invert_tl_ls_app)
        self.ui.pushButton_save_time_lapse_ls.clicked.connect(self.save_time_lapse_ls_app)
        self.ui.pushButton_show_time_lapse_ls.clicked.connect(self.show_time_lapse_ls_app)
        
        self.ui.label_status_ls.setText('Please load dataset')
        
# =============================================================================
#         setup: SHOW UI
# =============================================================================
        
        
        # show UI
        self.ui.show()
        # self.ui.raise_()
        
# =============================================================================
#         APP FUNCTIONS
# =============================================================================
        
        
    def file_browser(self):
        
        self.new_file_path, _ = QtWidgets.QFileDialog.getOpenFileName(directory = self.file_path, filter = '*.h5')
        # print(self.file_path)
        self.ui.lineEdit_file_path.setText(self.new_file_path)
        self.ui.pushButton_load_dataset.setEnabled(True)
    
    
    def _gather_params(self):
        
        params_from_ui = {'base': self.bases[self.ui.comboBox_base.currentIndex()],
                          'select_ROI': self.ui.checkBox_select_ROI.isChecked(),
                          'apply_denoise': self.ui.checkBox_denoise.isChecked(),
                          'X0': self.ui.spinBox_x0.value(),
                          'Y0': self.ui.spinBox_y0.value(),
                          'delta_x' : self.ui.spinBox_delta_x.value(),
                          'delta_y' : self.ui.spinBox_delta_y.value(),
                          'mu':self.ui.doubleSpinBox_mu.value(),
                          'lamda':self.ui.doubleSpinBox_lamda.value(),
                          'niter_out':self.ui.spinBox_niter_out.value(),
                          'niter_in':self.ui.spinBox_niter_in.value(),
                          'lsqr_niter':self.ui.spinBox_lsqr_niter.value(),
                          'lsqr_damp':self.ui.doubleSpinBox_lsqr_damp.value(),
                          'single_volume_time_index': self.ui.spinBox_t_frame_index.value(),
                          'save_label': self.ui.lineEdit_save_label.text(),
                          'plot_view': (1*self.ui.radioButton_plot_xz.isChecked() + 2*self.ui.radioButton_plot_yz.isChecked()),
                          'plot_mode':  self.plot_modes[self.ui.comboBox_plot_mode.currentIndex()],
                          'time_lapse_mode': self.time_lapse_modes[self.ui.comboBox_time_lapse_mode.currentIndex()],
                          'time_lapse_view': ( 1*self.ui.radioButton_xz.isChecked() + 2*self.ui.radioButton_yz.isChecked()),
                          'time_lapse_plane' : self.ui.spinBox_time_laps_plane.value(),
                          'time_lapse_save_label': self.ui.lineEdit_save_label_tl.text()
                          }
        return params_from_ui
    
    
    def _gather_params_ls(self):
        
        params_from_ui = {'select_ROI': self.ui.checkBox_select_ROI_ls.isChecked(),
                          'X0': self.ui.spinBox_x0_ls.value(),
                          'Y0': self.ui.spinBox_y0_ls.value(),
                          'delta_x' : self.ui.spinBox_delta_x_ls.value(),
                          'delta_y' : self.ui.spinBox_delta_y_ls.value(),
                          'mu':self.ui.doubleSpinBox_mu_ls.value(),
                          'lamda':self.ui.doubleSpinBox_lamda_ls.value(),
                          'niter_out':self.ui.spinBox_niter_out_ls.value(),
                          'niter_in':self.ui.spinBox_niter_in_ls.value(),
                          'lsqr_niter':self.ui.spinBox_lsqr_niter_ls.value(),
                          'lsqr_damp':self.ui.doubleSpinBox_lsqr_damp_ls.value(),
                          'single_volume_time_index': self.ui.spinBox_t_frame_index_ls.value(),
                          'save_label': self.ui.lineEdit_save_label_ls.text(),
                          'plot_view': (1*self.ui.radioButton_plot_xz_ls.isChecked() + 2*self.ui.radioButton_plot_yz_ls.isChecked()),
                          'plot_mode':  self.plot_modes[self.ui.comboBox_plot_mode_ls.currentIndex()],
                          'time_lapse_mode': self.time_lapse_modes[self.ui.comboBox_time_lapse_mode_ls.currentIndex()],
                          'time_lapse_view': ( 1*self.ui.radioButton_xz_ls.isChecked() + 2*self.ui.radioButton_yz_ls.isChecked()),
                          'time_lapse_plane' : self.ui.spinBox_time_laps_plane_ls.value(),
                          'time_lapse_save_label': self.ui.lineEdit_save_label_tl_ls.text()
                          }
        return params_from_ui
    
    
    def update_params(self):
        
        for key, val in self._gather_params().items():
            self.params[key] = val
            
    def update_params_ls(self):
        
        for key, val in self._gather_params_ls().items():
            self.ls_analyser.params[key] = val
            
    
    def load_file_path(self):
        self.file_path = self.new_file_path
        
        try:
            self.time_lapse = get_h5_attr(self.file_path, 'time_laps')[0] #TODO correct LAPS
        except:
            self.time_lapse = False
        try:
            self.params['time_frames_n'] = get_h5_attr(self.file_path, 'time_frames_n')[0]
            self.ui.spinBox_t_frame_index.setMaximum(self.params['time_frames_n'] -1)
        except:
            self.params['time_frames_n'] = None
            
        temp = 'time_frames_n'
        self.ui.label_time_lapse.setText(f'{self.time_lapse} ({self.params[temp] } time frame(s))')
        
        self.PosNeg = get_h5_attr(self.file_path, 'PosNeg')[0]
            
        self.ui.label_PosNeg.setText(f'{self.PosNeg}')
        self.subarray_hsize = get_h5_attr(self.file_path, 'subarray_hsize')[0]
        self.subarray_vsize = get_h5_attr(self.file_path, 'subarray_vsize')[0]
        self.ui.label_image_size.setText(f'{int(self.subarray_hsize):4d} x {int(self.subarray_vsize):4d} (px)')
        
        
        
        if self.new_file_path.find('coherent_SVIM') != -1 or self.new_file_path.find('Hadamard') != -1 :
            
            #choose tab
            self.ui.tabs.setCurrentWidget(self.ui.tab_coherent)
            self.ui.comboBox_base.setCurrentIndex(0)
            
            if self.new_file_path.find('Hadamard') != -1:
                self.ui.comboBox_base.setCurrentIndex(2)
                
            
            # init
            super().__init__(self.new_file_path, **self._gather_params())
            
            # enable 
            self.ui.groupBox_invert_single_volume.setEnabled(True)
            self.ui.label_status.setText('Ready to invert')
            self.ui.pushButton_save_inverted.setEnabled(False)
            self.ui.lineEdit_save_label.setEnabled(False)
            self.ui.label_save_label.setEnabled(False)
            self.ui.pushButton_show_inverted.setEnabled(False)
            self.ui.label_plot_view.setEnabled(False)
            self.ui.radioButton_plot_xy.setEnabled(False)
            self.ui.radioButton_plot_xz.setEnabled(False)
            self.ui.radioButton_plot_yz.setEnabled(False)
            self.ui.comboBox_plot_mode.setEnabled(False)
            self.ui.pushButton_save_inverted_time_lapse.setEnabled(False)
            self.ui.label_save_label_tl.setEnabled(False)
            self.ui.lineEdit_save_label_tl.setEnabled(False)
            self.ui.pushButton_show_time_lapse.setEnabled(False)
            
            self.ui.spinBox_t_frame_index.setEnabled(False)
            self.t_frame_index = 0
            if self.time_lapse:
                self.ui.groupBox_time_lapse.setEnabled(True)
                self.ui.label_t_frame_index.setEnabled(True)
                self.ui.spinBox_t_frame_index.setEnabled(True)
                self.ui.comboBox_time_lapse_mode.setEnabled(True)
                self.ui.label_tlview.setEnabled(True)
                self.ui.radioButton_xy.setEnabled(True)
                self.ui.radioButton_xz.setEnabled(True)
                self.ui.radioButton_yz.setEnabled(True)
                self.ui.pushButton_invert_time_lapse.setEnabled(True)
                
                
                
                
                
        elif self.new_file_path.find('DMD_light_sheet') != -1:
            # choose tab
            self.ui.tabs.setCurrentWidget(self.ui.tab_light_sheet)
            
            # init
            self.ls_analyser = DMD_light_sheet_analysis(self.new_file_path, **self._gather_params_ls())
            
            
            # enable ls
            
            self.ui.groupBox_volume_ls.setEnabled(True)
            self.ui.label_status_ls.setText('Ready to invert')
            self.ui.pushButton_save_volume_ls.setEnabled(False)
            self.ui.lineEdit_save_label_ls.setEnabled(False)
            self.ui.label_save_label_ls.setEnabled(False)
            self.ui.pushButton_save_time_lapse_ls.setEnabled(False)
            self.ui.label_save_label_tl_ls.setEnabled(False)
            self.ui.lineEdit_save_label_tl_ls.setEnabled(False)
            self.ui.pushButton_show_time_lapse_ls.setEnabled(False)
            
            
            self.ui.spinBox_t_frame_index.setEnabled(False)
            self.t_frame_index = 0
            if self.time_lapse:
                self.ui.groupBox_time_lapse_ls.setEnabled(True)
                self.ui.label_t_frame_index_ls.setEnabled(True)
                self.ui.spinBox_t_frame_index_ls.setEnabled(True)
                self.ui.comboBox_time_lapse_mode_ls.setEnabled(True)
                self.ui.label_tlview_ls.setEnabled(True)
                self.ui.radioButton_xy_ls.setEnabled(True)
                self.ui.radioButton_xz_ls.setEnabled(True)
                self.ui.radioButton_yz_ls.setEnabled(True)
                self.ui.pushButton_buils_time_lapse_ls.setEnabled(True)
            
        
        
        

    
    def show_im_raw_app(self):
        self.update_params()
        self.load_h5_file(self.t_frame_index)
        if self.select_ROI: self.setROI()
        if self.PosNeg: self.merge_pos_neg()
        self.show_im_raw()    
     

    def invert_volume_app(self):
        
        self.ui.label_status.setText('Inverting, please wait...')
        self.update_params()
        
        self.load_h5_file(self.t_frame_index)
        if self.select_ROI: self.setROI()
        if self.PosNeg: self.merge_pos_neg()
        
        
        if not self.denoise:
            # try:
                
            if self.params['base'] != 'hadam':
                self.choose_freq()
                self.p_invert()
            else:
                self.invert()
            # except:
                # print('Could not invert')
                    
        else:
            if self.params['base'] != 'hadam':
                self.choose_freq()
                
            self.invert_and_denoise3D_v2()  
            
            
        self.inverted = True
        self.ui.label_status.setText('Single Volume inversion completed')
        self.ui.pushButton_save_inverted.setEnabled(True)
        self.ui.lineEdit_save_label.setEnabled(True)
        self.ui.label_save_label.setEnabled(True)
        self.ui.pushButton_show_inverted.setEnabled(True)
        self.ui.label_plot_view.setEnabled(True)
        self.ui.radioButton_plot_xy.setEnabled(True)
        self.ui.radioButton_plot_xz.setEnabled(True)
        self.ui.radioButton_plot_yz.setEnabled(True)
        self.ui.comboBox_plot_mode.setEnabled(True)
        
        
        
        
        
    def save_inverted_app(self):
        
        self.update_params()
        self.save_inverted()
        self.ui.label_status.setText('Inverted Volume saved')
        
        
    def show_projections_app(self):
        
        self.update_params()
        
        # dmdPx_to_sample_ratio = 1.247 # (um/px)
        depth_z = (self.ROI_s_z * 1.247e-6 / self.image_inv.shape[0] )
        
        if self.params['plot_mode'] == 'ave':
            
            if self.params['plot_view'] == 0:   #xy
                title= f"Inverted image XY AVERAGE (base: {self.params['base']})"
                show_image(np.mean(self.image_inv, 0), title= title, ordinate = 'X', ascisse = 'Y', 
                           scale_ord = 0.65e-6, scale_asc = 0.65e-6)  
                
            elif self.params['plot_view'] == 1: #xz
                title= f"Inverted image XZ AVERAGE (base: {self.params['base']})"
                show_image(np.mean(self.image_inv, 2).transpose(), title= title, ordinate = 'X', ascisse = 'Z', 
                           scale_ord = 0.65e-6, scale_asc = depth_z )  
                
            elif self.params['plot_view'] == 2: #yz
                title= f"Inverted image YZ AVERAGE (base: {self.params['base']})"
                show_image(np.mean(self.image_inv, 1), title= title, ordinate = 'Z', ascisse = 'Y', 
                           scale_ord = depth_z, scale_asc = 0.65e-6 )  
                
        if self.params['plot_mode'] == 'max':
            
            if self.params['plot_view'] == 0:   #xy
                title= f"Inverted image XY MAX (base: {self.params['base']})"
                show_image(np.max(self.image_inv, 0), title= title, ordinate = 'X', ascisse = 'Y', 
                           scale_ord = 0.65e-6, scale_asc = 0.65e-6)  
                
            elif self.params['plot_view'] == 1: #xz
                title= f"Inverted image XZ MAX (base: {self.params['base']})"
                show_image(np.max(self.image_inv, 2).transpose(), title= title, ordinate = 'X', ascisse = 'Z', 
                           scale_ord = 0.65e-6, scale_asc = depth_z )  
                
            elif self.params['plot_view'] == 2: #yz
                title= f"Inverted image YZ MAX (base: {self.params['base']})"
                show_image(np.max(self.image_inv, 1), title= title, ordinate = 'Z', ascisse = 'Y', 
                           scale_ord = depth_z, scale_asc = 0.65e-6 )  
                
        elif self.params['plot_mode'] == 'stack':
            
            if self.params['plot_view'] == 0:   #xy
                title= f"Inverted image XY (base: {self.params['base']})"
                show_image(self.image_inv, title= title, ordinate = 'X', ascisse = 'Y', 
                           scale_ord = 0.65e-6, scale_asc = 0.65e-6)  
                
            elif self.params['plot_view'] == 1: #xz
                title= f"Inverted image XZ (base: {self.params['base']})"
                show_image(self.image_inv.transpose(2,1,0), title= title, ordinate = 'X', ascisse = 'Z', 
                           scale_ord = 0.65e-6, scale_asc = depth_z )  
                
            elif self.params['plot_view'] == 2: #yz
                title= f"Inverted image YZ (base: {self.params['base']})"
                show_image(self.image_inv.transpose(1,0,2), title= title, ordinate = 'Z', ascisse = 'Y', 
                           scale_ord = depth_z, scale_asc = 0.65e-6 ) 
                
        # show_image(image, title)      
    
        
    def invert_tl_app(self):
        
        self.update_params()
        self.invert_time_lapse()
        self.ui.label_status.setText('Time Lapse inversion completed')
        
        self.ui.pushButton_save_inverted_time_lapse.setEnabled(True)
        self.ui.label_save_label_tl.setEnabled(True)
        self.ui.lineEdit_save_label_tl.setEnabled(True)
        self.ui.pushButton_show_time_lapse.setEnabled(True)
        
        
    def save_time_lapse_app(self):
        
        self.update_params()
        self.save_time_lapse()
        self.ui.label_status.setText('Time Lapse saved')
        
# =============================================================================
#      LIGHT SHEET FUNCTIONS   
# =============================================================================
        
        
        
    def denoise_ls_app(self):
        
        self.update_params_ls()
        self.ls_analyser.load_h5_file()
        if self.select_ROI_ls : self.ls_analyser.setROI()
        print('denoise_ls')
        
        self.ls_analyser.denoise3D()
        
    def save_volume_ls_app(self):
        self.update_params_ls()
        if self.select_ROI_ls : self.ls_analyser.setROI()
        self.ls_analyser.save_volume()
        
    def show_projections_ls_app(self):
        print('show_projections_ls_app')
        
        self.update_params_ls()
        
        if self.ls_analyser.denoised == False:
            self.ls_analyser.load_h5_file()
            if self.select_ROI_ls : self.ls_analyser.setROI()
        
        
        # dmdPx_to_sample_ratio = 1.247 # (um/px)
        depth_z = (self.ls_analyser.ROI_s_z * 1.247e-6 / self.ls_analyser.image.shape[0] )
        
        if self.ls_analyser.params['plot_mode']:
            
            if self.ls_analyser.params['plot_view'] == 0:   #xy
                title= f"Light Sheet Volume: XY SUM"
                show_image(np.sum(self.ls_analyser.image, 0), title= title, ordinate = 'X', ascisse = 'Y', 
                           scale_ord = 0.65e-6, scale_asc = 0.65e-6)  
                
            elif self.ls_analyser.params['plot_view'] == 1: #xz
                title= f"Light Sheet Volume: XZ SUM"
                show_image(np.sum(self.ls_analyser.image, 2).transpose(), title= title, ordinate = 'X', ascisse = 'Z', 
                           scale_ord = 0.65e-6, scale_asc = depth_z )  
                
            elif self.ls_analyser.params['plot_view'] == 2: #yz
                title= f"Light Sheet Volume: YZ SUM"
                show_image(np.sum(self.ls_analyser.image, 1), title= title, ordinate = 'Z', ascisse = 'Y', 
                           scale_ord = depth_z, scale_asc = 0.65e-6 )  
                
        else:
            
            if self.ls_analyser.params['plot_view'] == 0:   #xy
                title= f"Light Sheet Volume: XY"
                show_image(self.ls_analyser.image, title= title, ordinate = 'X', ascisse = 'Y', 
                           scale_ord = 0.65e-6, scale_asc = 0.65e-6)  
                
            elif self.ls_analyser.params['plot_view'] == 1: #xz
                title= f"Light Sheet Volume: XZ"
                show_image(self.ls_analyser.image.transpose(2,1,0), title= title, ordinate = 'X', ascisse = 'Z', 
                           scale_ord = 0.65e-6, scale_asc = depth_z )  
                
            elif self.ls_analyser.params['plot_view'] == 2: #yz
                title= f"Light Sheet Volume: YZ"
                show_image(self.ls_analyser.image.transpose(1,0,2), title= title, ordinate = 'Z', ascisse = 'Y', 
                           scale_ord = depth_z, scale_asc = 0.65e-6 ) 
                
        
    def invert_tl_ls_app(self):
        print('invert_tl_ls_app')
        
        self.ui.pushButton_save_time_lapse_ls.setEnabled(True)
        self.ui.label_save_label_tl_ls.setEnabled(True)
        self.ui.lineEdit_save_label_tl_ls.setEnabled(True)
        self.ui.pushButton_show_time_lapse_ls.setEnabled(True)
      
    def save_time_lapse_ls_app(self):
        print('save_time_lapse_ls_app')
        
    def show_time_lapse_ls_app(self):
        print('show_time_lapse_ls_app')
    
          
    def exec_(self):
        return self.qtapp.exec_()


# =============================================================================
#           MAIN
# =============================================================================


if __name__ == '__main__':
    
    
    app = basic_app()
    app.setup()
    sys.exit(app.exec_())