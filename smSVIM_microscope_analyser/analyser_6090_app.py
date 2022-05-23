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
from analyser_transform_6090 import coherentSVIM_analysis

class basic_app(coherentSVIM_analysis):
    
    def __init__(self, argv = []):
        
        self.qtapp = QtWidgets.QApplication(argv)
        self.name = 'basic_app'
        self.qtapp.setApplicationName(self.name)
        self.qtapp.setStyle("mac")
        
      
        
    def setup(self):
        
        self.ui_filename = 'analyser_6090.ui'
        self.ui = uic.loadUi(self.ui_filename)
        
        # file path and load
        
        self.file_path = '/Users/marcovitali/Documents/Poli/tesi/ScopeFoundy/coherentSVIM/data/220511_rog_h2o2_first_round/220511_145138_coherent_SVIM_ (1).h5'
        self.ui.pushButton_file_browser.clicked.connect(self.file_browser)
        self.ui.pushButton_load_dataset.clicked.connect(self.load_file_path)
        
        # base selection
        
        self.bases = ['cos', 'sq']
        self.base = 'cos'
        self.ui.comboBox_base.activated.connect(self.change_base)
        
        # select ROI
        
        def enable_ROI():
            self.select_ROI = self.ui.checkBox_select_ROI.isChecked()
            if self.select_ROI:
                self.ui.widget_ROI_params.setEnabled(True)
            else:
                self.ui.widget_ROI_params.setEnabled(False)
        self.select_ROI = False
        self.ui.checkBox_select_ROI.stateChanged.connect(enable_ROI)
        
        
        def update_X0(): self.X0 = self.ui.spinBox_x0.value()
        def update_Y0(): self.Y0 = self.ui.spinBox_y0.value()
        def update_delta_x(): self.delta_x = self.ui.spinBox_delta_x.value()
        def update_delta_y(): self.delta_y = self.ui.spinBox_delta_y.value()
        
        self.ui.spinBox_x0.valueChanged.connect(update_X0)
        self.ui.spinBox_y0.valueChanged.connect(update_Y0)
        self.ui.spinBox_delta_x.valueChanged.connect(update_delta_x)
        self.ui.spinBox_delta_y.valueChanged.connect(update_delta_y)
        
        
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
        
        
        def update_mu(): self.mu = self.ui.doubleSpinBox_mu.value()
        def update_lamda(): self.lamda = self.ui.doubleSpinBox_lamda.value()
        def update_niter_out(): self.niter_out = self.ui.spinBox_niter_out.value()
        def update_niter_in(): self.niter_in = self.ui.spinBox_niter_in.value()
        def update_lsqr_niter(): self.lsqr_niter = self.ui.spinBox_lsqr_niter.value()
        def update_lsqr_damp(): self.lsqr_damp = self.ui.doubleSpinBox_lsqr_damp.value()
        
        self.ui.doubleSpinBox_mu.valueChanged.connect(update_mu)
        self.ui.doubleSpinBox_lamda.valueChanged.connect(update_lamda)
        self.ui.spinBox_niter_out.valueChanged.connect(update_niter_out)
        self.ui.spinBox_niter_in.valueChanged.connect(update_niter_in)
        self.ui.spinBox_lsqr_niter.valueChanged.connect(update_lsqr_niter)
        self.ui.doubleSpinBox_lsqr_damp.valueChanged.connect(update_lsqr_damp)
        
        # Invert single volume
        def update_t_frame_index():  self.t_frame_index = self.ui.spinBox_t_frame_index.value()
        self.ui.spinBox_t_frame_index.valueChanged.connect(update_t_frame_index)
        
        self.ui.pushButton_show_raw_im.clicked.connect(self.get_and_show_im_raw)
        self.ui.pushButton_invert.clicked.connect(self.get_and_invert)
        self.ui.pushButton_show_inverted.clicked.connect(self.show_inverted)
        self.ui.pushButton_show_inverted_xy.clicked.connect(self.show_inverted_xy)
        self.ui.pushButton_show_inverted_xz.clicked.connect(self.show_inverted_xz)
        
        
        # show UI
        self.ui.show()
        # self.ui.raise_()
        
    def file_browser(self):
        
        self.new_file_path, _ = QtWidgets.QFileDialog.getOpenFileName(directory = self.file_path, filter = '*coherent_SVIM*.h5')
        # print(self.file_path)
        self.ui.lineEdit_file_path.setText(self.new_file_path)
        self.ui.pushButton_load_dataset.setEnabled(True)
    
    def gather_params(self):
        
        params_from_ui = {'base': self.bases[self.ui.comboBox_base.currentIndex()],
                          'X0': self.ui.spinBox_x0.value(),
                          'Y0': self.ui.spinBox_y0.value(),
                          'delta_x' : self.ui.spinBox_delta_x.value(),
                          'delta_y' : self.ui.spinBox_delta_y.value(),
                          'mu':self.ui.doubleSpinBox_mu.value(),
                          'lamda':self.ui.doubleSpinBox_lamda.value(),
                          'niter_out':self.ui.spinBox_niter_out.value(),
                          'niter_in':self.ui.spinBox_niter_in.value(),
                          'lsqr_niter':self.ui.spinBox_lsqr_niter.value(),
                          'lsqr_damp':self.ui.doubleSpinBox_lsqr_damp.value()}
        return params_from_ui
    
    def load_file_path(self):
        super().__init__(self.new_file_path, self.gather_params())
        self.ui.groupBox_invert_single_volume.setEnabled(True)
        self.ui.label_status.setText('Ready to invert')
        self.ui.pushButton_save_inverted.setEnabled(False)
        self.ui.pushButton_show_inverted.setEnabled(False)
        self.ui.pushButton_show_inverted_xy.setEnabled(False)
        self.ui.pushButton_show_inverted_xz.setEnabled(False)
        
        try:
            self.time_lapse = get_h5_attr(self.file_path, 'time_laps')[0] #TODO correct LAPS
        except:
            self.time_lapse = False
        try:
            self.time_frames_n = get_h5_attr(self.file_path, 'approx_time_frames_n')[0]
        except:
            self.time_frames_n = None
            
        self.ui.label_time_lapse.setText(f'{self.time_lapse} ({self.time_frames_n} time frame)')
        self.PosNeg = get_h5_attr(self.file_path, 'PosNeg')[0]
        self.ui.label_PosNeg.setText(f'{self.PosNeg}')
        self.subarray_hsize = get_h5_attr(self.file_path, 'subarray_hsize')[0]
        self.subarray_vsize = get_h5_attr(self.file_path, 'subarray_vsize')[0]
        self.ui.label_image_size.setText(f'{int(self.subarray_hsize):4d} x {int(self.subarray_vsize):4d} (px)')
        
        self.ui.spinBox_t_frame_index.setEnabled(False)
        self.t_frame_index = 0
        if self.time_lapse:
            self.ui.groupBox_invert_time_lapse.setEnabled(True)
            self.ui.label_t_frame_index.setEnabled(True)
            self.ui.spinBox_t_frame_index.setEnabled(True)
     
    def change_base(self):
        base_index = self.ui.comboBox_base.currentIndex()
        self.base = self.bases[base_index]
        print(self.base)
    
    
    def get_and_show_im_raw(self):
        
        self.load_h5_file(self.t_frame_index)
        if self.select_ROI: self.setROI()
        if self.PosNeg: self.merge_pos_neg()
        self.show_im_raw()    
     

    def get_and_invert(self):
        
        self.ui.label_status.setText('Inverting, please wait...')
        
        self.load_h5_file(self.t_frame_index)
        if self.select_ROI: self.setROI()
        if self.PosNeg: self.merge_pos_neg()
        
        self.choose_freq()
        
        if not self.denoise:
            try:
                self.p_invert()
            except:
                print('Could not invert')
                    
        else:
            self.invert_and_denoise3D_v2()  
            
        self.inverted = True
        self.ui.label_status.setText('Inversion completed')
        self.ui.pushButton_save_inverted.setEnabled(True)
        self.ui.pushButton_show_inverted.setEnabled(True)
        self.ui.pushButton_show_inverted_xy.setEnabled(True)
        self.ui.pushButton_show_inverted_xz.setEnabled(True)
          
    def exec_(self):
        return self.qtapp.exec_()


if __name__ == '__main__':
    
    
    app = basic_app()
    app.setup()
    sys.exit(app.exec_())