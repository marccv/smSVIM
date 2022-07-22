from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget

import sys

from random import randint
import pyqtgraph as pg
from pyqtgraph import PlotWidget
import scipy.misc

class _new_image(PlotWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, window_counter):
        super().__init__()
        # layout = QVBoxLayout()
        # self.label = QLabel("Another Window % d" % randint(0,100))
        # layout.addWidget(self.label)
        # self.setLayout(layout)
        
        self.plot = self.getPlotItem()
        self.window_counter = window_counter


    def show_image(self, image, **kwargs):
                
        #default settings
        show_params = {'ordinate': 'X',
                       'ascisse' : 'Y',
                       'scale_ord' : 1,
                       'scale_asc' : 1}
        
        
        for key, val in kwargs.items():
            show_params[key] = val
        
        
        # app = pg.mkQApp()
        
        ordinate_text = f'<strong style="font-size: 20px;">{show_params["ordinate"]} axis</strong>'
        ascisse_text = f'<strong style="font-size: 20px;">{show_params["ascisse"]} axis</strong>'
        self.plot.setLabel(axis='left', text= ascisse_text,  units = 'm')
        self.plot.setLabel(axis='bottom', text=  ordinate_text, units = 'm')
        
        
        self.image_view = pg.ImageView(view = self.plot)
        self.img = self.image_view.getImageItem()
        tlabel = pg.InfLineLabel(self.image_view.timeLine, text="{value:.0f}")
        
        # cmap = cm.getFromMatplotlib('inferno') # prepare a linear color map
        # bar = pg.ColorBarItem( cmap=cm ) # prepare interactive color bar
        # # Have ColorBarItem control colors of img and appear in 'plot':
        # bar.setImageItem( img, insert_in=plot )
        # img.setColorMap(cmap)
        
        windowTitle = kwargs.pop("title", "ImageView")
        self.image_view.setWindowTitle(f'Fig {self.window_counter} - ' + windowTitle)
        self.image_view.setImage(image, scale = (show_params["scale_ord"], show_params["scale_asc"]), pos = (0, 0))
        # images.append(w)
        self.img.setAutoDownsample(True)
        self.image_view.show()
        
        
        def imageHoverEvent(event):
            """Show the position, pixel, and value under the mouse cursor.
            """
            if event.isExit():
                self.plot.setTitle("")
                return
            pos = event.pos()
            i, j  = pos.y(), pos.x()
            time_index = self.image_view.currentIndex
            # print(dir(pos))
            
            if len(image.shape) == 2:
            
                # i = int(np.clip(i, 0, image.shape[0] - 1))
                # j = int(np.clip(j, 0, image.shape[1] - 1))
                # val = image[i, j]
                # print(val)
                ppos = self.img.mapToParent(pos)
                x, y = ppos.x(), ppos.y()
                
            else:
                
                # i = int(np.clip(i, 0, image.shape[1] - 1))
                # j = int(np.clip(j, 0, image.shape[2] - 1))
                # val = image[time_index ,i, j]
                # print(val)
                ppos = self.img.mapToParent(pos)
                x, y = ppos.x(), ppos.y()
                
            self.plot.setTitle("pos: (%.1f, %.1f)um  -- pixel: (%d, %d, %d)" % (x*1e6, y*1e6, time_index, i, j), font = 15)
            
            # plot.setTitle("value: %d" % ( val))
    
        # Monkey-patch the image to use our custom hover function. 
        # This is generally discouraged (you should subclass ImageItem instead),
        # but it works for a very simple use like this. 
        self.img.hoverEvent = imageHoverEvent
        


class show_images_new_windows():

    def __init__(self):
        self.window_counter = 0
        self.window_list = []
        

    def show_new_image(self, image, **kwargs):
        
        self.window_counter += 1
        
        w = _new_image(self.window_counter)
        w.show_image(image, **kwargs)
        
        
        self.window_list.append(w)
        
        print('This is figure window number ', self.window_counter)
        
    def close_all(self):
        
        print('Closing all pg windows')
        
        if self.window_counter == 0: return
        
        for w in self.window_list:
            w.image_view.close()
            
        self.window_counter = 0
        self.window_list = []