# Import libraries

import os
import traitlets
from ipywidgets import *
from IPython.display import display, Javascript, clear_output

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline                    # Safe smooth interpolation
from scipy.signal import find_peaks
from scipy.signal import medfilt
from scipy import signal
from pybaselines import Baseline, utils
from scipy.integrate import simpson, trapz

from tkinter import Tk, filedialog

import peakutils
from peakutils.plot import plot as pplot

from tqdm import tqdm

import spe_read.spe_read as spe  

## Defining all functions

# DO NOT CHANGE anything - except the locaiton of your lamp data file ( it is in the i_corr function)

def data_cleaning(data,erp):
    
    # Code to remove cosmic rays
    #erp=100 # Set percentage error limit
    err=erp/100

    Ndatapts = len(data)
    ydata = data

    for i in range(0,Ndatapts): # This loop removes hot regions one pixel wide
        if i>0 and i<Ndatapts-1:
            if ydata[i]>(1+err)*ydata[i+1] and ydata[i]>(1+err)*ydata[i-1]:
                #print('Entered the loop 1')
                ydata[i]=(ydata[i-1]+ydata[i+1])/2

    for i in range(0,Ndatapts): # This loop removes hot regions up to 3 pixels wide
        if i>1 and i<Ndatapts-2 and ydata[i]>(1+err)*ydata[i+2] and ydata[i]>(1+err)*ydata[i-2]:
            #print('Entered the loop 3')
            ydata[i], ydata[i-1], ydata[i+1] = (ydata[i-2]+ydata[i+2])/2, (ydata[i]+ydata[i-2])/2, (ydata[i+2]+ydata[i])/2

    for i in range(0,Ndatapts): # This loop removes hot regions up to 5 pixels wide
        if i>4 and i<Ndatapts-5 and ydata[i]>(1+err)*ydata[i+5] and ydata[i]>(1+err)*ydata[i-5]:
            #print('Entered the loop 3')
            ydata[i], ydata[i-1], ydata[i-2], ydata[i-3], ydata[i-4], ydata[i+1], ydata[i+2], ydata[i+3], ydata[i+4] = (ydata[i-5]+ydata[i+5])/2, (ydata[i]+ydata[i-2])/2, (ydata[i-1]+ydata[i-3])/2, (ydata[i-2]+ydata[i-4])/2, (ydata[i-3]+ydata[i-5])/2, (ydata[i+2]+ydata[i])/2, (ydata[i+3]+ydata[i+1])/2, (ydata[i+4]+ydata[i+2])/2, (ydata[i+5]+ydata[i+3])/2

    for i in range(0,Ndatapts): # This loop removes dead regions one pixel wide
        if i>0 and i<Ndatapts-1 and ydata[i]<(1-err)*ydata[i+1] and ydata[i]<(1-err)*ydata[i-1]:
            #print('Entered the loop 4')
            ydata[i]=(ydata[i-1]+ydata[i+1])/2

    for i in range(0,Ndatapts): # This loop removes dead regions up to 3 pixels wide
        if i>1 and i<Ndatapts-2 and ydata[i]<(1-err)*ydata[i+2] and ydata[i]<(1-err)*ydata[i-2]:
            #print('Entered the loop 5')
            ydata[i], ydata[i-1], ydata[i+1] =(ydata[i-2]+ydata[i+2])/2, (ydata[i]+ydata[i-2])/2, (ydata[i+2]+ydata[i])/2
            
    return ydata

def subtract_bsl(x,y,hf):
    
    baseline_fitter = Baseline(x, check_finite=False)
    base = baseline_fitter.mor(y, half_window=hf)[0]
    
    return base

def filter_savgol(data,k_size,p_order):
    
    data[np.isnan(data)] = 0
    
    sav_filt_data=signal.savgol_filter(data,k_size,p_order)
    
    return sav_filt_data

def filter_median(data,k_size):
    
    filt_data = medfilt(data,kernel_size=k_size)
    
    return filt_data



def slicing_func(x,y,yi,start_x,end_x):
    start = np.argmin(abs(x-start_x))
    end = np.argmin(abs(x-end_x))
    #print(start,end)
    x_cut = x[start:end]
    y_cut = y[start:end]
    yi_cut = yi[start:end]
    
    return x_cut, y_cut, yi_cut, start, end

def _3Lorentzian(x, amp1, cen1, wid1, amp2,cen2,wid2, amp3,cen3,wid3):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2)) +\
            (amp2*wid2**2/((x-cen2)**2+wid2**2)) +\
                (amp3*wid3**2/((x-cen3)**2+wid3**2))

def norm_func(x, y, do_normalize):
    
    if do_normalize == 'relative':
        norm_intens_data = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif do_normalize == 'scale':
        norm_intens_data = y / np.max(y)
    elif do_normalize == 'area':
        norm_intens_data = y/np.trapz(y,x)
    else:
        norm_intens_data = y
    
    return norm_intens_data

def remove_cosmic_rays(selectFiles,batching,batch_size,average,final_file_name='-cosmic_ray-rmvd.csv'):
    """Remove cosmic rays from y_all by taking median within each batch.

    Args:
        y_all (list): List of numpy arrays to process.
        batch_size (int): Size of each batch.

    Returns:
        numpy.ndarray: Median array after processing each batch.
    """
    y_all = []

    fdata = selectFiles.files
    
    if len(fdata) < 3:
        print("Please select at least 3 files to remove cosmic rays")
        
    head_i, tail_i = os.path.split(fdata[0])
    for idx, file in enumerate(fdata):
    
        x_data, y_data, metadata = DataReader(file_name=file).read_file()
        y_all.append(y_data)
    # Convert the numpy arrays to column vectors
    column_vectors = [np.expand_dims(arr, axis=1) for arr in y_all]

    # Concatenate the column vectors along the axis=1
    concatenated_array = np.concatenate(column_vectors, axis=1)
    median_array = concatenated_array
    
    median_batches = []
    
    if batching==True:
        # Get the total number of elements and batches
        # print(concatenated_array.shape)
        total_elements = concatenated_array.shape[1]
        total_batches = total_elements // batch_size

        # Calculate the median within each batch
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_concat = np.concatenate(column_vectors[:][start_idx:end_idx], axis=1)
            batch_median = np.median(batch_concat, axis=1)
            median_batches.append(batch_median)

            # Save each batch_median to a CSV file
            batch_filename = f'-batch_median_{i+1}.csv'
            total_median_data = np.column_stack((x_data, batch_median))
            np.savetxt(os.path.join(head_i,tail_i[:-13]+batch_filename), total_median_data, delimiter=',')
    
        if total_batches == 1: average = False
        
    # Concatenate the median batches into a single array
    median_array = np.median(concatenated_array, axis=1)
    total_median_data = np.column_stack((x_data, median_array))
    
    # Save the final median_array to a CSV file
    np.savetxt(os.path.join(head_i,tail_i[:-13]+final_file_name), total_median_data, delimiter=',')
    
    if batching == True & average==True:
        averaged_data = np.average(median_batches, axis=0)
        total_average_data = np.column_stack((x_data, averaged_data))
        np.savetxt(os.path.join(head_i,tail_i[:-13]+"-averaged_data.csv"), total_average_data, delimiter=',')
        
    elif batching == False & average==True:
        averaged_data = np.average(median_batches, axis=0)
        total_average_data = np.column_stack((x_data, averaged_data))
        np.savetxt(os.path.join(head_i,tail_i[:-13]+"-averaged_data.csv"), total_average_data, delimiter=',')

    # Return the final median array
    return x_data, median_array

def data_averaging(selectFiles,average,batching, batch_size, final_file_name='-averaged_data.csv'):
    
    fdata = selectFiles.files
    
    if len(fdata) < 2:
        print("Please select at least 2 files to average")
        
    head_i, tail_i = os.path.split(fdata[0])
    y_all = []
    for idx, file in enumerate(fdata):
    
        x_data, y_data, metadata = DataReader(file_name=file).read_file()
        y_all.append(y_data)
    
    column_vectors = [np.expand_dims(arr, axis=1) for arr in y_all]

    # Concatenate the column vectors along the axis=1
    concatenated_array = np.concatenate(column_vectors, axis=1)
    average_array = concatenated_array

    average_batches = []

    if batching==True:
        # Get the total number of elements and batches
        # print(concatenated_array.shape)
        total_elements = concatenated_array.shape[1]
        total_batches = total_elements // batch_size

        # Calculate the median within each batch
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_concat = np.concatenate(column_vectors[:][start_idx:end_idx], axis=1)
            batch_average = np.average(batch_concat, axis=1)
            average_batches.append(batch_average)

            # Save each batch_median to a CSV file
            batch_filename = f'-batch_average_{i+1}.csv'
            total_median_data = np.column_stack((x_data, batch_average))
            np.savetxt(os.path.join(head_i,tail_i[:-13]+batch_filename), total_median_data, delimiter=',')

    
    averaged_data = np.average(y_all, axis=1)
    total_average_data = np.column_stack((x_data, averaged_data))
    np.savetxt(os.path.join(head_i,tail_i[:-13]+final_file_name), total_average_data, delimiter=',')
    
    return x_data, averaged_data

def mean_f_wvl(x,y,meanf_method):
    
    lambda_f = None
    
    if meanf_method == 'area':
        lambda_f = np.sum(x*y**2) /np.sum(y**2)
    elif meanf_method == 'sum':
        lambda_f = np.sum(x*y)/np.sum(x)
    elif meanf_method == 'integrate':
        lambda_f = simpson(x*y,x)/simpson(y,x)
    elif meanf_method == 'energy':
        x = 1240/x
        y = (1240/(x**2))*y
        e_f = simpson(x*y,x)/simpson(y,x)
        lambda_f = 1240/e_f
        
    return lambda_f
                
def _1Lorentzian(x, amp1, cen1, wid1):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2))

def ram2nm(x,laser_source):
    return 10**7/(10**7/laser_source - x)

def nm2ram(x,laser_source):
    return 10**7*(1/laser_source - 1/x)

def nm2eV(x):
    return 1232/x

def eV2nm(x):
    return x/1232

def plot_meanf_data(laser_Pow, meanf_df, ax2, plot_title):
    for laser_power in laser_Pow:
        #power_str = str(laser_power)
        df2 = meanf_df.iloc[list(meanf_df["Power"]==laser_power)]
        meanf_val = df2['Mean Wvl'].mean()
        sme_val = df2['Mean Wvl'].std()/df2['Mean Wvl'].count()
        ax2.errorbar(laser_power,meanf_val,yerr=sme_val, marker='s',color='k',markersize=10,capsize=6,barsabove=True,ecolor='k',elinewidth=2,capthick=1)
        
    ax2.set_title(plot_title)
    ax2.set_xlabel('1020 nm laser power')
    ax2.set_ylabel('Mean f wavelength')
    #ax2.set_xlim()
    #ax2.set_ylim()
    
    #ax2.set_yticks([])
    #ax2.tick_params(axis='y',which='both',left='off',labelleft='off')

    #ax2.legend(bbox_to_anchor=(1, 1),title='')
    
    return ax2


def i_corr_cleaning(iFiles,do_baseline_subtraction,do_median_filtering,do_data_cleaning,do_savgol_filtering,do_conversion,do_normalize,do_peak_finding,p_order,k_size,erp,k_size_savgol,p_order_savgol):
    idata = iFiles.files

    for i in tqdm(range(len(idata))):
        
        idata_name = idata[i]
        print(idata_name)

        head_i, tail_i = os.path.split(idata[i])

        ixdata, iydata = DataReader(file_name=idata[i]).read_file()

        if do_median_filtering == 'y':
            filt_idata = filter_median(iydata, k_size)
        else:
            filt_idata = iydata

        if do_baseline_subtraction == 'y':
            base = subtract_bsl(filt_idata, p_order)
        else:
            base = 0

        bsl_subt_idata = filt_idata - base

        if do_data_cleaning == 'y':
            cleaned_idata = data_cleaning(bsl_subt_idata, erp)
        else:
            cleaned_idata = bsl_subt_idata

        if do_savgol_filtering == 'y':
            savgol_filt_idata = filter_savgol(cleaned_idata, k_size_savgol,p_order_savgol)
        else:
            savgol_filt_idata = cleaned_idata

        ixval, iyval = ixdata, savgol_filt_idata

        iyval_new = np.nan_to_num(iyval,nan=0,posinf=0,neginf=0)
        i_datamatrix = np.column_stack((ixval,iyval_new)) # Compiles corrected data into a new matrix
        i_save = idata_name[:-4]+"_cleaned.txt" # Create filename for new data
        print(i_save)
        head_is, tail_is = os.path.split(i_save)
        np.savetxt(i_save, i_datamatrix) # Save new data
        
        #ax1.plot(ixval, iyval_new,label=tail_is) # Code line to plot the curves
    # ax1.set_title("Intensity Correction files")
    # ax1.set_xlabel(x_label)
    #ax1.set_ylabel('Intensity (counts)')

    # ax1.set_xlim(3000,4000)
    # ax1.set_ylim(1.5e-5,3.5e-5)

    # ax1.legend(bbox_to_anchor=(0.8, 0.8), loc='center', title="File Name")

    # ax1.grid(color = 'green', which='both', linestyle = '-.', linewidth = 0.5, alpha=0.4)
        
    # plt.show()
    # plt.savefig()

if __name__ == "__main__":
    pass