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

import numpy as np

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
        
        self.ui_filename = 'analyser_6090_tabs.ui'
        self.ui = uic.loadUi(self.ui_filename)
        
        # file path and load
        
        self.file_path = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/220523_cuma_fluo_test/220523_110615_coherent_SVIM_diff_300ul_transp.h5'
        self.ui.pushButton_file_browser.clicked.connect(self.file_browser)
        self.ui.pushButton_load_dataset.clicked.connect(self.load_file_path)
        
        self.params = {}
        
        # base selection
        
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
        
        self.ui.pushButton_save_inverted.clicked.connect(self.save_inverted_app)
        self.ui.pushButton_show_inverted.clicked.connect(self.show_projections_app)
        
        
        
        # time lapse section
        self.time_lapse_modes = ['sum', 'plane']
        def change_tl_mode():
            if self.ui.comboBox_time_lapse_mode.currentIndex() == 0:
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
        # show UI
        self.ui.show()
        # self.ui.raise_()
        
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
                          'plot_sum':  self.ui.checkBox_plot_sum.isChecked(),
                          'time_lapse_mode': self.time_lapse_modes[self.ui.comboBox_time_lapse_mode.currentIndex()],
                          'time_lapse_view': ( 1*self.ui.radioButton_xz.isChecked() + 2*self.ui.radioButton_yz.isChecked()),
                          'time_lapse_plane' : self.ui.spinBox_time_laps_plane.value(),
                          'time_lapse_save_label': self.ui.lineEdit_save_label_tl.text()
                          }
        return params_from_ui
    
    
    def update_params(self):
        
        for key, val in self._gather_params().items():
            self.params[key] = val
    
    def load_file_path(self):
        super().__init__(self.new_file_path, self._gather_params())
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
        self.ui.checkBox_plot_sum.setEnabled(False)
        self.ui.pushButton_save_inverted_time_lapse.setEnabled(False)
        self.ui.label_save_label_tl.setEnabled(False)
        self.ui.lineEdit_save_label_tl.setEnabled(False)
        self.ui.pushButton_show_time_lapse.setEnabled(False)
        
        
        if self.new_file_path.find('Hadamard') != -1:
            self.ui.comboBox_base.setCurrentIndex(2)
        elif self.new_file_path.find('DMD_light_sheet') != -1:
            print('LIGHT_SHEET')
        
        
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
        self.ui.checkBox_plot_sum.setEnabled(True)
        
        
        
        
        
    def save_inverted_app(self):
        
        self.update_params()
        self.save_inverted()
        self.ui.label_status.setText('Inverted Volume saved')
        
        
    def show_projections_app(self):
        
        self.update_params()
        
        dmdPx_to_sample_ratio = 1.247 # (um/px)
        aspect_xz = (self.ROI_s_z * dmdPx_to_sample_ratio / self.image_inv.shape[0] )/0.65
        
        if self.params['plot_sum']:
            
            if self.params['plot_view'] == 0: #xy
                image = np.sum(self.image_inv, 0)
            elif self.params['plot_view'] == 1: #xz
                image = np.repeat(np.sum(self.image_inv, 2).transpose(), aspect_xz, 1)
            elif self.params['plot_view'] == 2: #yz
                image = np.repeat(np.sum(self.image_inv, 1), aspect_xz, 0)
        else:
            
            if self.params['plot_view'] == 0:
                image = self.image_inv
            elif self.params['plot_view'] == 1:
                image= np.repeat(self.image_inv.transpose(2,1,0), aspect_xz, 2)
            elif self.params['plot_view'] == 2:
                image = np.repeat(self.image_inv.transpose(1,0,2), aspect_xz, 1)
                
        pg.image(image, title= f"Inverted image (base: {self.params['base']})")      
    
        
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
        
          
    def exec_(self):
        return self.qtapp.exec_()


if __name__ == '__main__':
    
    
    app = basic_app()
    app.setup()
    sys.exit(app.exec_())