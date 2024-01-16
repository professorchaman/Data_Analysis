import numpy as np
import pandas as pd
import os
import spe2py as spe

class DataReader():
    
    def __init__(self, file_name):
        self.file_name = file_name
        
    def read_file(self):
        
        fname = os.path.splitext(self.file_name)[0]
        ext = os.path.splitext(self.file_name)[1]
        
        metadata = None
        
        ext_dict = {".csv" : (",",0), ".txt" : ("\t",0), ".dat" : (None,0), ".spe" : ()}
        
        if ext == ".spe":
            x, y, metadata = self.read_spe()
            return x, y, metadata

        data = pd.read_csv(self.file_name, skiprows=ext_dict[ext][1], sep=ext_dict[ext][0],engine='python',header=None)
        x = data[0].to_numpy()
        y = data[1].to_numpy()
        
        if ext == ".dat":
            col_len = len(data.columns)
            
            y = np.zeros(shape=(len(x),col_len-1))

            for k in range(0,col_len-1):
                y[:,k] = data.iloc[:,k+1].values
        
        return x, y, metadata

    def read_xlsx(self):
        return pd.read_excel(self.file_name)

    def read_json(self):
        return pd.read_json(self.file_name)
    
    def read_spe(self):
        spe_obj = spe.spe_reader(file_name=self.file_name)
        
        file_data = []
        metadata_dict = dict()
        
        metadata_footer = spe_obj.spe_files.footer

        grove, cw = spe_obj.get_grating_info(metadata_footer)
        et = spe_obj.get_exposure_time(metadata_footer)
        acc_val, acc_method = spe_obj.get_accumulation_info(metadata_footer)
        
        x = spe_obj.spe_files._get_wavelength()
        
        file_data[:] = spe_obj.spe_files.data[:][0][0]
        
        metadata_dict = {
            "grating" : str(grove) + " g/mm",
            "center_wavelength" : str(cw) + " nm",
            "exposure_time" : et + " msec",
            "accumulation_value" : acc_val,
            "accumulation_method" : acc_method
        }
        
        return x, file_data, metadata_dict
        # return NotImplementedError 
