# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:54:06 2021

@author: Chaman Gupta
"""

'''
Run these commands in powershell or command prompt

pip install PeakUtils
pip install pyqt5

'''

import sys
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import pandas as pd

from PyQt5.QtWidgets import (QFileDialog,QPushButton,QCheckBox,QSpinBox,QLabel)
from PyQt5.QtCore import QDir, Qt
import numpy as np
import peakutils
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline
from scipy import signal


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        plt.rcParams["font.family"] = "arial"
        #plt.style.use('seaborn')
        
        #fig.tight_layout()
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
        
        self.display_width = 100
        self.display_height = 100
        
        self.setGeometry(700, 300, 200, 200)
        
        

        # Create the maptlotlib FigureCanvas object, 
        # which defines a single set of axes as self.axes.
        
        self.button1 = QPushButton('Import Data File')
        self.button1.clicked.connect(self.get_text_file)
        
        self.comboBox = QtWidgets.QComboBox()
        self.comboBox.setObjectName("comboBox")
        
        self.themes = ['bmh', 'classic', 'dark_background', 'fast', 
		'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright',
		 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 
		 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook',
		 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk',
		 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn',
		 'Solarize_Light2', 'tableau-colorblind10']
        
        self.theme_val = self.themes[7]
        
        self.comboBox.addItems(self.themes)
        
        self.comboBox.currentIndexChanged['QString'].connect(self.on_draw)
		
        #self.comboBox.currentIndexChanged['QString'].connect(self.Update)
        
        self.resize(600,600)
        self.sc = MplCanvas(self)
        self.sc.axes.clear()
        
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.sc, self)
        
        self.grid_cb = QCheckBox("Show &Grid")
        self.grid_cb.setChecked(False)
        self.grid_cb.stateChanged.connect(self.on_draw) #int
        #self.horizontalLayout.addWidget(self.comboBox)
        
        self.hbox = QtWidgets.QHBoxLayout()
        
        self.i_label = QLabel("Select Lamp PL for intensity correction")
        self.button5 = QPushButton('Import Lamp File')
        self.button5.clicked.connect(self.get_lamp_file)
        
        self.button6 = QPushButton('Intensity Correction')
        self.button6.clicked.connect(self.i_corr)
        
        vbox4 = QtWidgets.QVBoxLayout()
        vbox4.addWidget(self.i_label)
        vbox4.addWidget(self.button5)
        vbox4.addWidget(self.button6)
        
        self.sp2label = QLabel("Enter error % for Data Cleaning")
        self.spin2 = QSpinBox()
        self.spin2.setRange(0, 100)
        self.spin2.setSingleStep(1)
        self.spin2.setValue(5)
        
        
        self.button2 = QPushButton('Data Cleaning')
        self.button2.clicked.connect(self.data_cleaning)
        
        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.addWidget(self.sp2label)
        vbox2.addWidget(self.spin2)
        vbox2.addWidget(self.button2)
        
        # creating spin box
        self.sp1label = QLabel("Enter polynomial order for Baseline subtraction")
        self.spin1 = QSpinBox()
        self.spin1.setRange(1, 7)
        self.spin1.setSingleStep(2)
        self.spin1.setValue(3)

        #self.spin1.valueChanged.connect(self.show_result)   
        
        self.button3 = QPushButton('Baseline Subtraction')
        self.button3.clicked.connect(self.subtract_bsl)
        
        vbox1 = QtWidgets.QVBoxLayout()
        vbox1.addWidget(self.sp1label)
        vbox1.addWidget(self.spin1)
        vbox1.addWidget(self.button3)
        
        self.sp3label = QLabel("Enter kernel box size for median filtering")
        self.spin3 = QSpinBox()
        self.spin3.setRange(1, 7)
        self.spin3.setSingleStep(2)
        self.spin3.setValue(3)

        #self.spin1.valueChanged.connect(self.show_result)   
        
        self.button4 = QPushButton('Median Filtering')
        self.button4.clicked.connect(self.filter_median)
        
        vbox3 = QtWidgets.QVBoxLayout()
        vbox3.addWidget(self.sp3label)
        vbox3.addWidget(self.spin3)
        vbox3.addWidget(self.button4)
        
        self.button5 = QPushButton('Savgol Smoothening')
        self.button5.clicked.connect(self.filter_savgol)

        self.sp4label = QLabel("Enter kernel box for Savgol smoothening")
        self.spin4 = QSpinBox()
        self.spin4.setRange(1, 131)
        self.spin4.setSingleStep(2)
        self.spin4.setValue(51)
        
        self.sp5label = QLabel("Enter p_order for Savgol smoothening")
        self.spin5 = QSpinBox()
        self.spin5.setRange(1, 10)
        self.spin5.setSingleStep(1)
        self.spin5.setValue(1)
        
        vbox5 = QtWidgets.QVBoxLayout()
        vbox5.addWidget(self.sp4label)
        vbox5.addWidget(self.spin4)
        vbox5.addWidget(self.sp5label)
        vbox5.addWidget(self.spin5)
        vbox5.addWidget(self.button5)
        
        
        
        for w in [self.grid_cb]:
            self.hbox.addWidget(w)
            self.hbox.setAlignment(w, Qt.AlignVCenter)
        self.hbox.addLayout(vbox5)
        self.hbox.addLayout(vbox4)
        self.hbox.addLayout(vbox3)
        self.hbox.addLayout(vbox2)
        self.hbox.addLayout(vbox1)
            
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.comboBox)
        vbox.addWidget(self.button1)
        vbox.addWidget(toolbar)
        vbox.addWidget(self.sc)
        
        vbox.addLayout(self.hbox)
        
        self.widget = QtWidgets.QWidget()
        self.widget.setLayout(vbox)
        self.setCentralWidget(self.widget)
        
        self.x = None
        self.y = None


        self.show()
        
    def get_lamp_file(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setFilter(QDir.Files)
        
        self.lamp_file, _ = QFileDialog.getOpenFileName(self, 'Open Data File', r"<Default dir>", "Data files (*.csv *.txt)")
        
        header_list = ["x-data", "y-data"]
        
        lamp_df = pd.read_csv(self.lamp_file,names=header_list,sep=None,engine='python')
        
        #df.plot(ax=self.sc.axes)
        
        self.df_lamp_x = lamp_df['x-data'].to_numpy()
        self.df_lamp_y = lamp_df['y-data'].to_numpy()
        
        self.lamp_x = self.df_lamp_x
        self.lamp_y = self.df_lamp_y
    
    def get_text_file(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setFilter(QDir.Files)
        
        self.file_name, _ = QFileDialog.getOpenFileName(self, 'Open Data File', r"<Default dir>", "Data files (*.csv *.txt)")
        
        self.data_plot()
     
    def data_plot(self):

        header_list = ["x-data", "y-data"]
        
        df = pd.read_csv(self.file_name,names=header_list,sep=None,engine='python')
        
        #df.plot(ax=self.sc.axes)
        
        self.df_num_x = df['x-data'].to_numpy()
        self.df_num_y = df['y-data'].to_numpy()
        
        self.x = self.df_num_x
        self.y = self.df_num_y
    
        self.on_draw()

        self.update()
        
    def on_draw(self):
        """ Redraws the figure
        """
        plt.style.use(self.theme_val)
        self.xd = self.x
        self.yd = self.y
        
        self.sc.axes.clear()    
        self.sc.axes.grid(self.grid_cb.isChecked())
        #self.axes.grid(self.grid_cb.isChecked())
        
        self.sc.axes.plot(self.xd,self.yd)
        
        self.sc.draw()
        
        #self.x = []
        #self.y = []
    
    def data_cleaning(self):
    
        # Code to remove cosmic rays
        erp=self.spin2.value() # Set percentage error limit
        err=erp/100
        
        ydata = self.y

        Ndatapts = len(ydata)
        
    
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
                
        self.y = ydata
        self.on_draw()
        
        return ydata
    
    def subtract_bsl(self):
    
   
        ydata = np.asarray(self.y)
        p_order = self.spin1.value()
        
        base = peakutils.baseline(ydata,p_order)
        
        ydata = ydata - base
        self.y = ydata
        
        self.on_draw()
   
    def filter_savgol(self):
        
        ydata = np.asarray(self.y)
        
        ydata[np.isnan(ydata)] = 0
        
        k_size_sav = self.spin4.value()
        p_order_sav = self.spin5.value()
        
    
        sav_filt_data = signal.savgol_filter(ydata,k_size_sav,p_order_sav)
    
        self.y = sav_filt_data
        
        self.on_draw()  
   
    def filter_median(self):
        
        ydata = np.asarray(self.y)
        k_size = self.spin1.value()
    
        filt_data = medfilt(ydata,kernel_size=k_size)
        
        self.y = filt_data
        
        self.on_draw()
    
    def i_corr(self):
    
        ## Important note to change the location of this .txt file below to where it is in your computer.
        calibstd = np.loadtxt(r"G:\Shared drives\Pauzauskie Team Drive\CG\Scripts\030410638_HL-2000-CAL_2014-01-15-14-09_VISEXT1EXT2_FIB.txt")
        xcalib = calibstd[:,0]
        ycalib = calibstd[:,1]
        
        #print(len(ycalib))
    
        #filenum = 21 # I dont think this actually does anything. It was already there when I received the fil
        
        x,y = self.x,self.y
        #x=np.asarray(x, dtype=np.float32)
        #y=np.asarray(y, dtype=np.float32) #Splits up data into X and Y
        #if len(fdark)!=0: #Does background correction if chosen
            #xdark, ydark = masterRead(fdark,0)
            #xdark=np.asarray(xdark, dtype=np.float32)
            #ydark=np.asarray(ydark, dtype=np.float32)
            #y=y-ydark
        HglampFunc = CubicSpline(xcalib,ycalib)
        hglampI = HglampFunc(x) # Create interpolation of true lamp spectrum
        #print(len(hglampI))
        #print('flamp shape = ' + str(len(flamp)))
        hglampdata_x, hglampdata_y = self.df_lamp_x,self.df_lamp_y # Split true lamp spectra into x and y
        #x=np.asarray(x, dtype=np.float32)
        #y=np.asarray(y, dtype=np.float32) #Splits up data into X and Y
        #print(len(hglampdata_y))
        #if len(flampbase)!=0: # Substract lamp background if selected
           # xlampdark, ylampdark = masterRead(flampbase,0, start=start, end=end)
            #xlampdark=np.asarray(xlampdark, dtype=np.float32)
            #ylampdark=np.asarray(ylampdark, dtype=np.float32)
            #hglampdata_y = hglampdata_y-ylampdark
        ICF = hglampI/(hglampdata_y) # Creates ratio of true lamp spectra to real lamp data, ICF = Intensity Correction Factor
        ynew = (y)*ICF # multiplies real data by intensity correction factor
        datamatrix = np.column_stack((x,ynew)) # Compiles corrected data into a new matrix
        #savename = self.file_name[:-4]+"_calib.txt" # Create filename for new data
        #np.savetxt(savename,datamatrix) # Save new data
        
        self.y = ynew
        
        self.on_draw()
       
        

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Breeze')
    w = MainWindow()
    
    app.exec_()