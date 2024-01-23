import os
import numpy as np
import pandas as pd
from data_reader.datareader import DataReader
from scipy.interpolate import CubicSpline

def i_corr(flamp,f,i,save_correction=True):
    
    """
    This function takes in a lamp spectrum and a data spectrum and corrects the data spectrum for the intensity of the lamp spectrum.
    The lamp spectrum is a .txt file of the true lamp spectrum. The data spectrum is a .txt file of the data spectrum.
    The function outputs the corrected data spectrum as a .txt file.
    
    Parameters
    ----------
    flamp : str
        The file name of the lamp spectrum.
    f : str
        The file name of the data spectrum.
    i : int
        The index of the data spectrum in the list of data spectra.
    save_correction : bool, optional
        If True, saves the corrected data spectrum as a .txt file. The default is True.
        
    Returns
    -------
    x : array
        The x-axis data of the corrected data spectrum.
    y : array
        The y-axis data of the corrected data spectrum.
    intensity_corrected_data : array
        The corrected data spectrum.
    
    """
    
    ## Important note to change the location of this .txt file below to where it is in your computer.
    calibration_standard = np.loadtxt(r"G:\Shared drives\Pauzauskie Team Drive\CG\Scripts\030410638_HL-2000-CAL_2014-01-15-14-09_VISEXT1EXT2_FIB.txt")
    
    calibration_x_data = calibration_standard[:,0]
    calibration_y_data = calibration_standard[:,1]

    x, y, _ = DataReader(file_name=f[i]).read_file()

    hg_lamp_fit = CubicSpline(calibration_x_data,calibration_y_data)
    hg_lamp_interp = hg_lamp_fit(x) # Create interpolation of true lamp spectrum
    
    _, hg_lampdata_y, _ = DataReader(file_name=flamp[0]).read_file() # Split true lamp spectra into x and y

    icf = hg_lamp_interp/(hg_lampdata_y) # Creates ratio of true lamp spectra to real lamp data, icf = Intensity Correction Factor

    intensity_corrected_data = (y)*icf # multiplies real data by intensity correction factor

    intensity_corrected_data = np.nan_to_num(intensity_corrected_data,nan=0,posinf=0,neginf=0)

    if save_correction == True:
        datamatrix = np.column_stack((x,intensity_corrected_data)) # Compiles corrected data into a new matrix
        np.savetxt(f[i][:-4]+"_icf.txt", datamatrix, delimiter=",") # Saves corrected data as a .txt file

    return x, y, intensity_corrected_data