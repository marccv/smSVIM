

from ScopeFoundry import BaseMicroscopeApp
name = 'coherentSVIM_App'

class coherent_SVIM_App(BaseMicroscopeApp):
    
    def setup(self):
            
        from TexasInstrumentsDMD_ScopeFoundry.DMDHardware import TexasInstrumentsDmdHW
        self.add_hardware(TexasInstrumentsDmdHW(self))
        
        from Hamamatsu_ScopeFoundry.CameraHardware import HamamatsuHardware
        self.add_hardware(HamamatsuHardware(self))
                
        from PI_ScopeFoundry.PICoilStageHardware import PIStageNew
        self.add_hardware(PIStageNew(self))    
        
        from Shutter_ScopeFoundry.shutter_hw import ShutterHW
        self.add_hardware(ShutterHW(self))  
        
        from Hamamatsu_ScopeFoundry.CameraMeasurement import HamamatsuMeasurement
        self.add_measurement(HamamatsuMeasurement(self))
        
        from smSVIM_Microscope.coherentSVIM_Measurement import coherentSvimMeasurement
        self.add_measurement(coherentSvimMeasurement(self))
        
        from smSVIM_Microscope.coherentSVIM_Hadamard_Measurement import coherentSvim_Hadamard_Measurement
        self.add_measurement(coherentSvim_Hadamard_Measurement(self))
       
        from smSVIM_Microscope.dmd_light_sheet_Measurement import DMD_light_sheet_measurement
        self.add_measurement(DMD_light_sheet_measurement(self))
        
        
        self.ui.show()
        self.ui.activateWindow()
        
        
      
    
if __name__ == '__main__':
            
    import sys
    app = coherent_SVIM_App(sys.argv)
    
    
    ################## for debugging only ##############
    # app.settings_load_ini(".\\settings\\settings0.ini")
    # for hc_name, hc in app.hardware.items():
    #     hc.settings['connected'] = True
    ####################################################    
        
    sys.exit(app.exec_())

        
