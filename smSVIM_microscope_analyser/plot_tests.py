#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:26:01 2022

@author: marcovitali
"""


import sys
from qtpy import QtWidgets, uic
import qtpy.QtCore
import pyqtgraph as pg
from get_h5_data import get_h5_dataset, get_h5_attr
import h5py
from analyser_transform_6090 import coherentSVIM_analysis

import sys
import os
import h5py
import pyqtgraph as pg
from qtpy.QtWidgets import QApplication

import numpy as np

import scipy.misc


images = []
plots = []
QAPP = None




def show_image(*args, **kargs):
    """
    Create and return an :class:`ImageView <pyqtgraph.ImageView>` 
    Will show 2D or 3D image data.
    Accepts a *title* argument to set the title of the window.
    All other arguments are used to show data. (see :func:`ImageView.setImage() <pyqtgraph.ImageView.setImage>`)
    """
    app = pg.mkQApp()
    plot = pg.PlotItem()
    plot.setLabel(axis='left', text='Y axis', units = 'm')
    plot.setLabel(axis='bottom', text='<strong style="font-size: 20px;">X axis</strong>', units = 'm')
    
    
    
    w = pg.ImageView(view = plot)
    img = w.getImageItem()
    
    tlabel = pg.InfLineLabel(w.timeLine, text="{value:.0f}")
    # print(dir(tlabel))
    windowTitle = kargs.pop("title", "ImageView")
    w.setWindowTitle(windowTitle)
    w.setImage(scale = (2, 1), pos = (0, 0), *args)
    images.append(w)
    
    w.show()
    
    
    
    
    
    
    
    return w, tlabel


if __name__ == "__main__" :
    
    
    image_color = scipy.misc.face()
    
    image = image_color.transpose(2,1,0)[:,:,:].copy()
    
    t_label = show_image(image, title = 'ciao')
    
    if sys.flags.interactive != 1 or not hasattr(qtpy.QtCore, 'PYQT_VERSION'):
        QApplication.exec_()
    # sys.exit ( "End of test")