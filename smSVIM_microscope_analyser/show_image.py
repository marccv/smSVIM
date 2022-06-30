#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 19:39:16 2022

@author: marcovitali
"""

import pyqtgraph as pg
# import cmap as cm



def show_image(image,  **kwargs):
    
    images = []
    plots = []
    QAPP = None
    
    #default settings
    show_params = {'ordinate': 'X',
                   'ascisse' : 'Y',
                   'scale_ord' : 1,
                   'scale_asc' : 1}
    
    
    for key, val in kwargs.items():
        show_params[key] = val
    
    
    app = pg.mkQApp()
    plot = pg.PlotItem()
    
    ordinate_text = f'<strong style="font-size: 20px;">{show_params["ordinate"]} axis</strong>'
    ascisse_text = f'<strong style="font-size: 20px;">{show_params["ascisse"]} axis</strong>'
    plot.setLabel(axis='left', text= ascisse_text,  units = 'm')
    plot.setLabel(axis='bottom', text=  ordinate_text, units = 'm')
    
    
    w = pg.ImageView(view = plot)
    img = w.getImageItem()
    tlabel = pg.InfLineLabel(w.timeLine, text="{value:.0f}")
    
    # cmap = cm.getFromMatplotlib('inferno') # prepare a linear color map
    # bar = pg.ColorBarItem( cmap=cm ) # prepare interactive color bar
    # # Have ColorBarItem control colors of img and appear in 'plot':
    # bar.setImageItem( img, insert_in=plot )
    # img.setColorMap(cmap)
    
    windowTitle = kwargs.pop("title", "ImageView")
    w.setWindowTitle(windowTitle)
    w.setImage(image, scale = (show_params["scale_ord"], show_params["scale_asc"]), pos = (0, 0))
    images.append(w)
    w.show()
    
    
    def imageHoverEvent(event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            plot.setTitle("")
            return
        pos = event.pos()
        i, j  = pos.y(), pos.x()
        time_index = w.currentIndex
        # print(dir(pos))
        
        if len(image.shape) == 2:
        
            # i = int(np.clip(i, 0, image.shape[0] - 1))
            # j = int(np.clip(j, 0, image.shape[1] - 1))
            # val = image[i, j]
            # print(val)
            ppos = img.mapToParent(pos)
            x, y = ppos.x(), ppos.y()
            
        else:
            
            # i = int(np.clip(i, 0, image.shape[1] - 1))
            # j = int(np.clip(j, 0, image.shape[2] - 1))
            # val = image[time_index ,i, j]
            # print(val)
            ppos = img.mapToParent(pos)
            x, y = ppos.x(), ppos.y()
            
        plot.setTitle("pos: (%.1f, %.1f)um  -- pixel: (%d, %d, %d)" % (x*1e6, y*1e6, time_index, i, j), font = 15)
        
        # plot.setTitle("value: %d" % ( val))

    # Monkey-patch the image to use our custom hover function. 
    # This is generally discouraged (you should subclass ImageItem instead),
    # but it works for a very simple use like this. 
    img.hoverEvent = imageHoverEvent
    
    
    return w




if __name__ == "__main__" :
    
    from qtpy.QtWidgets import QApplication

    import sys
    import scipy.misc
    import qtpy.QtCore
    
    
    image_color = scipy.misc.face()
    
    image = image_color.transpose(2,1,0)[:,:,:].copy()
    
    t_label = show_image(image, title = 'ciao')
    
    if sys.flags.interactive != 1 or not hasattr(qtpy.QtCore, 'PYQT_VERSION'):
        QApplication.exec_()
    # sys.exit ( "End of test")
