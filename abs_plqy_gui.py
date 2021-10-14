"""Absolute PLQY GUI

This script (abs_plqy_gui.py) is for a graphical interface used to measure
the absolute photoluminescence quantum yield (PLQY) of samples placed in an
integrating sphere and excited with a laser/LED. The integrating sphere (Labsphere)
is connected to the spectrometer (Ocean optics QE PRO) with a fiber optics cable
and is callibrated by Ocean Optics, who provided the calibration file.
For the excitation source, a 405 nm laser (LDM405, Thorlabs) or a 415 nm
LED source (M415F3, Thorlabs) was used. The laser/LED was partially refected
using an angled quartz substrate onto a Si detector (818-UV/DB, Newport)
that was connected to a digital multimeter (34410A, Agilent). The spectrometer
was connected to the controlling computer by direct USB connection, while
the multimeter was connected using a GPIB-USB-HS connector (National Instruments).

This GUI enables both the data collection and analysis for determination of the 
absolute PLQY of your material. The data collection involves collecting a dark
spectrum (no light or sample), blank (e.g. cuvette with solvent) and sample
(e.g. cuvette with solvent and quantum dots). The power of the light is also 
measured by the external Si detector and recorded; this allows the fluctuations
and drifting of the light intensity to be catered for.

The data analysis involves integrating the normalized spectrum of both
the blank and sample data at both the light excitation region and the PL region.
Then, the PLQY is calculated as the ratio of photons emitted to photons
absorbed. The standard deviation is also calculated using multiple rounds
of measurement. Note, however, that this does not consider systemic error 
caused by calibration accuraccies, etc. 
 

To use this GUI, you will need to install the seabreeze and pymeasure packages,
which are open-source packages. If you are using Anaconda, you can install 
it via conda-forge:

# install via conda
conda install -c conda-forge seabreeze
conda install -c conda-forge pymeasure

Seabreeze and pymesure documentation:
https://python-seabreeze.readthedocs.io/en/latest/
https://pymeasure.readthedocs.io/en/latest/

@author: Jia-Ahn Pan
"""

from seabreeze.spectrometers import Spectrometer
from pymeasure.instruments.agilent import Agilent34410A
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as fd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,NavigationToolbar2Tk)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import os
from textwrap import wrap
from datetime import datetime
import threading
import queue
import time


spec_serial_number = "QEP00140" 
multimeter_port = "GPIB::4"
plt.rcParams['savefig.dpi'] = 500 #resolution of saved graph

lbound_low = 400 #excitation bounds in nm
lbound_high = 410
fbound_low = 500 #PL bounds in nm
fbound_high = 710

integration_time = 1000 # integration time in us
max_counts = 170000 #depends on spectrometer
number_subruns = 1
number_meas = 2
blank_name = "blank"
sample_name = "sample"

text_instructions = """
Overview
This GUI is for the measurement of the absolute PL quantum yield using an integrating sphere.\
 It controls both an Ocean Optics QEPRO and an Agilent 34410A mulimeter that is connected to a Si detector.

Initialization
1. Click "Initialize instruments" to initialize the spectrometer and multimeter (make sure they are both on).
2. Close the enclosure. Set the integration time and number of subruns. To determine a good integration time, turn the live view on\
 by pressing the button. Increase the integration time until it sits below the maximum intensity line.
3. With the laser off and no blank/sample inserted, click "Collect dark spectrum" to collect the dark spectrum.
4. Click "Calibration file to load the calibration file, e.g."QEP00140_cus_20190620_1555.IRRADCAL".

Data collection
1. Click on "Saving folder" and select the folder to save the data and analysis in
2. Select number of measurements to take. Each measurement run involves collecting the number\
 of subruns specified and averaging their intensities. 
3. Insert your blank (e.g. cuvette with solvent).
4. Turn on the blue LED/laser.
5. Specify the blank name and click "Collect blank(s)"\
 Each measurement will each be saved in the data file\
 as "meas_0", "meas1", etc. and the average of the runs will also be given.\
 The power of the laser will be measured throughout each run and it's value is also saved.
6. Repeat steps 3-5, but with the sample.

Data analysis
1. If you have just collected the data, the blank and sample file should be automatically populated.\
 If you want to specify the blank and sample file, click "Blank file" or "Sample file" and select the relevant file.
2. Enter the bounds for your excitation and for your PL
3. Click "Analyze and plot" to analyze and plot your data. If needed, change your bounds and click it again.
4. Click "Save analysis file & plot" to save the plot and analysis data in the same directory.
5. Turn off the laser/LED after measurements are done.

~by Jia-Ahn Pan   
"""


def collect_intensities(): 
    """Data collection with spectrometer and multimeter.
    
    Collects n number of subruns based on user input. For each subrun, also
    collects power data from the external Si detector by calling the function
    "measure_34410a_current_average", which runs on a new thread. Returns the 
    spectrum and power intensities as a dataframe.
    """

    number_of_subruns = int(initialize_frame.entry_subruns.get()) # number of subruns to average over
    data_array = []
    power_mean_array = []
    for n in range(0, number_of_subruns):
        initialize_frame.label_collection_progress.config(text="Collecting " + str(n+1) + " of " 
                                                  + str(number_of_subruns) + "...")
        e = threading.Event() # make event to synchronize power detection with spectrometer
        threading.Thread(name="power measure thread", target=measure_34410a_current_average,
                         args=(e,)).start() #start power measurement on new thread
        initialize_frame.spec.features["data_buffer"][0].clear() 
        data = initialize_frame.spec.intensities(correct_dark_counts=True) #collect data from spectrometer
        e.set() # trigger event to stop power measurement
        time.sleep(0.2) #wait for power measurement to finish
        power = initialize_frame.current_uA_mean #power data from "measure_34410a_current_average(e)" function
        data_array.append(data)
        power_mean_array.append(power)
        
    initialize_frame.label_collection_progress.config(text="Collection done. ")
    data_average = np.mean(data_array, axis=0)
    power_average = np.mean(power_mean_array, axis=0)
    initialize_frame.queue.put(data_average) #save data in queue so that main thread can read and plot

    return data_average, power_average


def measure_34410a_current_average(e):
    """Measure the DC current from the multimeter in sync with spectrometer
    
    The DC current from the multimeter is proportional to the intensity of light
    from the Si detector. As long as the event, e, is not set, it collects data.
    Once e is set, which indicates that the spectrometer is done, it breaks out
    of the loop and averages all the values

    """
    power_array = []
    while not e.isSet():
        current_uA = initialize_frame.multimeter.current_dc*1e6
        initialize_frame.label_multimeter_current_value.config(text=round(current_uA,4))
        power_array.append(current_uA)
    initialize_frame.current_uA_mean =  np.mean(power_array)
    

def collect_data_loop_thread():
    """Starts a separate data collection thread so that GUI does not freeze
    Calls "collect_data_loop"
    """
    threading.Thread(name="collection thread", target=collect_data_loop).start()


def collect_data_loop(): 
    """Main loop for data collection options
    
    Waits for the user to presss a button for dark spectrum, blank spectrum,
    sample spectrum  or live view. Pressing these buttons turns the relevant
    boolean to True which calls the relevant function. If dark, blank, or
    sample is collected, turn all bool to False to prevent accidental 
    back-to-back measurements when multiple buttons have been prressed.
    
    After 10 ms, calls "collect_data_loop_thread" to start a new thread that
    calls back "collect_data_loop"
    """
    
    if (initialize_frame.collect_dark_bool == True or
        collect_data_frame.collect_blank_bool == True or
        collect_data_frame.collect_sample_bool == True):
        
        if initialize_frame.collect_dark_bool == True:
            initialize_frame.dark_spec()
            initialize_frame.label_collection_progress.config(text="Dark collection done!")
        elif collect_data_frame.collect_blank_bool == True:
            collect_data_frame.collect_blank()
            initialize_frame.label_collection_progress.config(text="Blank collection done!")
        elif collect_data_frame.collect_sample_bool == True:
            collect_data_frame.collect_sample()
            initialize_frame.label_collection_progress.config(text="Sample collection done!")
        time.sleep(0.2)
        # change all to False to prevent back-to-back measurement
        initialize_frame.collect_dark_bool = False
        collect_data_frame.collect_blank_bool = False
        collect_data_frame.collect_sample_bool = False 
        
    elif initialize_frame.live_view_bool == True:
        collect_intensities()
    
    initialize_frame.after(10, collect_data_loop_thread) #repeat loop after 10 ms
       

class InitializeFrame(ttk.LabelFrame):
    """Initialization frame (top left)
    
    Frame for initialization of instruments and setting parameters
    """
    
    def __init__(self, container):
        #initialization
        super().__init__(container, padding=(5,5), 
                         text="(A) Initialization")
                         
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=5)
        self.columnconfigure(2, weight=5)
        
        self.__widgets() 
        self.__plot_figure()
        
        self.grid(column=0, row=0,ipadx=1, ipady=1, padx=5, pady=5, sticky=tk.N)
        
 
    def __widgets(self):
        #code for widgets (button, labels, entry)
        self.button_init_spec = ttk.Button(self, text="Initialize instruments",
                                           command=self.initialize_inst)
        self.button_init_spec.grid(column=0, row=0, sticky=tk.W)
        
        self.label_init_spec = ttk.Label(self, text="Please initialize spectrometer and multimeter")
        self.label_init_spec.grid(column=1, row=0, columnspan=2, sticky=tk.W)
        
        # live view button
        self.live_view_bool = False
        self.button_live_view = ttk.Button(self, text="Live view on/off",
                                          command=self.live_view_bool_switch)
        self.button_live_view.grid(column=2, row=4, rowspan=1, sticky=tk.NS)
        
        #saturation point
        self.max_intensity_bool = True
        self.button_max_intensity = ttk.Button(self, text="Max int. line on/off",
                                               command=self.max_intensity_bool_switch)
        self.button_max_intensity.grid(column=2, row=5, sticky=tk.NS)
        
        #collection progress
        self.label_collection_progress = ttk.Label(self, text="",
                                                   foreground="teal")
        self.label_collection_progress.grid(column=0, row=2)
       
        # Integration time label and entry
        self.label_int_time = ttk.Label(self, text="Integration time (ms): ", background=None,
                                anchor="center")
        self.label_int_time.grid(column=0, row=4,sticky=tk.W) 
        
        self.int_time_str = tk.IntVar(value=integration_time)
        self.int_time_str.trace_add("write", self.set_int_time)
        self.entry_int_time = ttk.Entry(self, width=8, textvariable=self.int_time_str)
        self.entry_int_time.grid(column=1, row=4, sticky=tk.W)
       
        
        #Number of subruns
        self.label_subruns = ttk.Label(self, text="Number of subruns:", background=None,
                                anchor="center")
        self.label_subruns.grid(column=0, row=5, sticky=tk.W) 
        
        self.entry_subruns = ttk.Entry(self, width=8, background="red")
        self.entry_subruns.insert(0, number_subruns)
        self.entry_subruns.grid(column=1, row=5, sticky=tk.W)
        
        # #Collect dark spectrum
        self.collect_dark_bool = False
        self.button_dark = ttk.Button(self, text="Collect dark spectrum",
                                      command=self.collect_dark_bool_true)
        self.button_dark.grid(column=0, row=6, rowspan=1, sticky=tk.EW)
        
        self.label_dark = ttk.Label(self, text="Please collect dark spectrum")
        self.label_dark.grid(column=1, row=6, columnspan=2, sticky=tk.EW)
        
        # Wavelength calibration file
        self.button_calibration = ttk.Button(self, text="Calibration file",
                                             command=self.calibration_file)
        self.button_calibration.grid(column=0, row=7, sticky=tk.EW)
        
        self.label_calibration = ttk.Label(self, text="Load calibration file")
        self.label_calibration.grid(column=1, row=7, columnspan=2, sticky=tk.W)
        
        # Power-meter current
        self.label_multimeter_current = ttk.Label(self, text="Si photodiode current (uA): ")
        self.label_multimeter_current.grid(column=0, row=3, pady=(3,15))


    def __plot_figure(self):
        #plot blank figure
        self.fig = Figure(figsize=(6,4), dpi=60, tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        
        #display canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=1, columnspan=4) 
        
        self.toolbar_frame = tk.Frame(master=self)
        self.toolbar_frame.grid(column=1, row=2, columnspan=2,
                                 sticky=tk.E)
        self.toolbar = Toolbar(self.canvas, self.toolbar_frame).update()
        
    def initialize_inst(self):
        #Button command to initialize instruments
        self.init_multimeter()
        self.init_spec()
    
    def init_multimeter(self):
        #Initialize multimeter connected to Si detector
        self.multimeter = Agilent34410A(multimeter_port)
        self.multimeter.reset()
        self.multimeter.clear()
        self.current_uA = round(self.multimeter.current_dc*1e6, 4)
        
        self.label_multimeter_current_value = ttk.Label(self, text=self.current_uA) #display current
        self.label_multimeter_current_value.grid(column=1, row=3, sticky=tk.W, pady=(3,15))

    
    def init_spec(self):
        #initialize the spectrometer
        self.spec = Spectrometer.from_serial_number(spec_serial_number) 
        self.label_init_spec.config(text=self.spec) #show spectrometer name
        self.set_int_time() #set default integration time

        self.df = pd.DataFrame({"wavelength": self.spec.wavelengths()}) #save wavelength points
        self.df["delta_x"] = np.gradient(self.df["wavelength"]) #save wavelength spacing
        
        self.queue = queue.Queue() # queue used to transfer data between threads
        collect_data_loop_thread() #begin loop for data collection on new thread
        self.update_graph() #begin loop for updating graph
    
    
    def set_int_time(self, *args):
        #set/change integration time
        self.spec.integration_time_micros(self.int_time_str.get()*1e3)        
        
        
    def update_graph(self):
        # update graph for live view or dark/blank/sample measurement
        # if queue is not empty (data has been collected), then get and plot values
        #refreshes every 10 ms
        
        if not self.queue.empty():
            data = self.queue.get()
            self.ax.clear()    
            self.ax.plot(self.df["wavelength"], data)
            if self.max_intensity_bool == True:
                self.ax.axhline(max_counts, color="grey", linestyle="dashed")
            self.canvas.draw() 
        self.after(10, self.update_graph)   
        
        
    def live_view_bool_switch(self):
        if self.live_view_bool == False:
            self.live_view_bool = True
        else:
            self.live_view_bool = False
            
    def max_intensity_bool_switch(self):
        if self.max_intensity_bool == False:
            self.max_intensity_bool = True
        else:
            self.max_intensity_bool = False
                
    def collect_dark_bool_true(self):
        self.collect_dark_bool = True
        
        
    def dark_spec(self):
        # collect dark spectra and display collection time
        self.df["dark"], _ = collect_intensities()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        self.label_dark.config(text="Calibrated at " + current_time)
        
        
    def calibration_file(self):
        #load calibration file and display
        self.calib_file_dir= fd.askopenfilename(title="Select callibration file", filetypes = (("Text files", "*.IRRADCAL"),("All files","*.*")))
        self.df["irradcal"] = pd.read_csv(self.calib_file_dir, sep=("\t"), 
                  skiprows=8, usecols=[1])
        self.label_calibration.config(text=os.path.basename(self.calib_file_dir))
               
        
        
class CollectDataFrame(ttk.LabelFrame):
    """Collection data frame (bottom left)
    
    Frame for data collection
    """
    def __init__(self, container):
        super().__init__(container, padding=(5,5), text="(B) Data Collection")
        
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=12)
        
        self.__widgets()
        
        #show the frame
        self.grid(column=0, row=1,ipadx=1, ipady=1, padx=5, pady=5, sticky=tk.NSEW)
        
    def __widgets(self):
        #select folder
        self.button_folder = ttk.Button(self, text="Saving folder", 
                                       command=self.saving_folder, background=None)
        self.button_folder.grid(column=0, row=0, sticky=tk.NSEW)
                
        self.label_folder = ttk.Label(self, text="Select folder to save in")
        self.label_folder.grid(column=1, row=0, sticky=tk.W)
       
        
        #Number of measurements
        self.label_number_meas = ttk.Label(self, text="Number of measurements:",
                                           wraplength=100)
        self.label_number_meas.grid(column=0, row=1, sticky=tk.W)
        
        self.entry_number_meas = ttk.Entry(self, width=8)
        self.entry_number_meas.insert(0, number_meas)
        self.entry_number_meas.grid(column=1, row=1, sticky=tk.W)
        
        #Blank name and collect bleck and blank button
        self.label_blank = ttk.Label(self, text="Blank name:")
        self.label_blank.grid(column=0, row=2, pady=(15,0),sticky=tk.W)
        
        self.entry_blank = ttk.Entry(self, width=16)
        self.entry_blank.insert(0, blank_name)
        self.entry_blank.grid(column=1, row=2, pady=(15,0), sticky=tk.W)
        
        self.collect_blank_bool = False
        self.button_blank = ttk.Button(self, text="Collect blank(s)",
                                       command=self.collect_blank_bool_true)
        self.button_blank.grid(column=0, row=3, columnspan=2, sticky=tk.NSEW)
        
        #Sample name and collect sample button
        self.label_sample = ttk.Label(self, text="Sample name:")
        self.label_sample.grid(column=0, row=4, pady=(15,0), sticky=tk.W)
        
        self.entry_sample = ttk.Entry(self, width=16)
        self.entry_sample.insert(0, sample_name)
        self.entry_sample.grid(column=1, row=4, pady=(15,0), sticky=tk.W)
        
        self.collect_sample_bool = False
        self.button_sample = ttk.Button(self, text="Collect sample(s)",
                                        command=self.collect_sample_bool_true)
        self.button_sample.grid(column=0, row=5, columnspan=2, sticky=tk.NSEW)
    
    def saving_folder(self):
        #select folder to save in
        self.folder_path = fd.askdirectory(title="Select file(s) for blank")
        self.label_folder.config(text=self.folder_path)
        
    def collect_multi(self):
        """Collection of multiple measurements for sample/blank
      
        For each measurment, subtract dark spectrum, multiply by calibration
        curve, divide by wavelength spacing (delta_x) and multiply by wavelength.
        We divide by the wavelength spacing in order to get Joules/nm. Then,
        multiply by wavelength to convert into photon counts/nm. Also,
        save the power from Si detector. Output both the spectrum and power
        intensity as dataframes.
        """
        number_of_meas = int(collect_data_frame.entry_number_meas.get())

        inten_dict = {}
        power_dict = {}
    
        for n in range(number_of_meas):
            intensities, power_meas = collect_intensities()
            inten_dict["meas_" + str(n)] = (intensities - initialize_frame.df["dark"])*\
                initialize_frame.df["irradcal"]/initialize_frame.df["delta_x"]*\
                initialize_frame.df["wavelength"]/1000     
            power_dict["meas" + str(n) + "_power"] = [power_meas]    
            
    
        df_multi = pd.concat(inten_dict, axis=1) #make into dataframe
        df_multi["mean_photon_counts"] = df_multi.mean(axis=1)
        df_multi.insert(0, "wavelength", initialize_frame.df["wavelength"])
        
        df_power = pd.DataFrame(power_dict)
        df_power["laser_power_mean_uA"] = df_power.mean(axis=1)

        return df_multi, df_power
    
    
    def collect_blank_bool_true(self):
        self.collect_blank_bool = True
        
    def collect_sample_bool_true(self):
        self.collect_sample_bool = True
       
    def collect_blank(self):
        """Collect and save blank data

        """
        self.df_blank, self.df_blank_power = self.collect_multi()
        current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
  
        blank_path_name = f"{self.folder_path}/{current_datetime}_{self.entry_blank.get()}.csv"
        
        self.df_blank_power.to_csv(blank_path_name, mode="a", index=False) #write power df
        
        with open(blank_path_name, 'a', newline="") as f:
            f.write("\n")
        
        self.df_blank.to_csv(blank_path_name, mode="a", index=False)
        
        #load analysis path with this data
        analyze_data_frame.blank_file_dir = blank_path_name
        analyze_data_frame.df_blank = pd.read_csv(analyze_data_frame.blank_file_dir, skiprows=3)
        analyze_data_frame.df_blank_power = pd.read_csv(analyze_data_frame.blank_file_dir
                                                        , skiprows=lambda x:x not in [0,1])

        analyze_data_frame.blank_name = os.path.basename(analyze_data_frame.blank_file_dir)
        analyze_data_frame.label_blank.config(text=analyze_data_frame.blank_name)
        
    def collect_sample(self):
        """Collect and save sample data

        """
        self.df_sample, self.df_sample_power = self.collect_multi()
        current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        # sample_path_name = self.folder_path + r"/" + current_datetime + r"_" +\
        #     self.entry_sample.get() + r".csv"
        sample_path_name = f"{self.folder_path}/{current_datetime}_{self.entry_sample.get()}.csv"
       
        self.df_sample_power.to_csv(sample_path_name, mode="a", index=False) #write power df
        
        with open(sample_path_name, 'a', newline="") as f:
            f.write("\n")
        
        self.df_sample.to_csv(sample_path_name, mode="a", index=False)
        
        #load analysis path with sample data
        analyze_data_frame.sample_file_dir = sample_path_name 
        analyze_data_frame.df_sample = pd.read_csv(analyze_data_frame.sample_file_dir, skiprows=3)
        analyze_data_frame.df_sample_power = pd.read_csv(analyze_data_frame.sample_file_dir
                                                        , skiprows=lambda x:x not in [0,1])
        analyze_data_frame.sample_name = os.path.basename(analyze_data_frame.sample_file_dir)
        analyze_data_frame.label_sample.config(text=analyze_data_frame.sample_name)
            
        
        
class AnalyzeDataFrame(ttk.LabelFrame):
    """Analysis data frame (right)
    
    Frame for PLQY calculation
    """
    def __init__(self, container):
        super().__init__(container, padding=(5,5),
                         text="(C) Photoluminescence Quantum Yield Calculator")
          
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)
        self.columnconfigure(2, weight=3)
        self.columnconfigure(3, weight=3)
        self.columnconfigure(4, weight=8)
        
        self.__widgets()
        self.__plot_figure()
        
        #show the frame
        self.grid(column=1, row=0, rowspan=2, ipadx=1, ipady=1, padx=5, 
                  pady=5, sticky=tk.NSEW)
        
    def __widgets(self):
         # button for blank and sample
        self.button_blank = tk.Button(self, text="Blank file",
                                      command=self.import_blank)
        self.button_sample = tk.Button(self, text= "Sample file",
                                       command=self.import_sample)
        
        self.button_blank.grid(row=0, column=0, sticky=tk.EW)
        self.button_sample.grid(row=1, column=0, sticky=tk.EW)
        
        # Text for blank and sample
        self.label_blank = tk.Label(self, text = "Please select blank(s)")
        self.label_sample = tk.Label(self, text = "Please select sample(s)")
        
        self.label_blank.grid(row=0,column=1, columnspan=3, sticky=tk.W)
        self.label_sample.grid(row=1,column=1,columnspan=3, sticky=tk.W)
        
        #input laser and PL range
        self.label_laser_low = tk.Label(self, text="Excitation lower bound (nm):")
        self.label_laser_high = tk.Label(self, text="Excitation upper bound (nm):") 
        self.label_pl_low = tk.Label(self, text="PL lower bound (nm):")
        self.label_pl_high = tk.Label(self, text="PL upper bound (nm):")
        
        self.label_laser_low.grid(row=2,column=0, sticky=tk.W)
        self.label_laser_high.grid(row=2,column=2, sticky=tk.W)
        self.label_pl_low.grid(row=3,column=0, sticky=tk.W)
        self.label_pl_high.grid(row=3,column=2, sticky=tk.W)
        
        self.entry_laser_low = tk.Entry(self, width=5)
        self.entry_laser_high = tk.Entry(self, width=5)
        self.entry_pl_low = tk.Entry(self, width=5)
        self.entry_pl_high = tk.Entry(self, width=5)
        
        self.entry_laser_low.insert(0, lbound_low)
        self.entry_laser_high.insert(0, lbound_high)
        self.entry_pl_low.insert(0, fbound_low) 
        self.entry_pl_high.insert(0, fbound_high)
        
        self.entry_laser_low.grid(row=2,column=1, sticky=tk.W)
        self.entry_laser_high.grid(row=2,column=3, sticky=tk.W)
        self.entry_pl_low.grid(row=3,column=1, sticky=tk.W)
        self.entry_pl_high.grid(row=3,column=3, sticky=tk.W)
        
        
        #analyze_data_button
        self.button_analyze_data = tk.Button(self, text="Analyze and plot",
                                             command=self.calculate_plqy)
        self.button_analyze_data.grid(column=0, row=5, sticky=tk.EW)
        
        #save analysis file
        self.button_save_analysis = tk.Button(self, text="Save analysis file & plot",
                                              command=self.save_analysis)
        self.button_save_analysis.grid(column=1, row=5, sticky=tk.EW)
    
        
        #instruction pop-up window
        self.button_instruct = tk.Button(self, text="Instructions",
                                         command=self.instructions,
                                         foreground="white",
                                         background = "green")
        self.button_instruct.grid(column=4, row=0, columnspan=2,rowspan=2,
                                  sticky=tk.NSEW)
    
    def __plot_figure(self):
        #show empty graph
        self.fig = Figure(figsize=(6,4), dpi=100, tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
      
        self.plot_analysis = self.fig.add_subplot(111)
     
        self.canvas.draw()        
        self.canvas.get_tk_widget().grid(column=0, row=4,
                                          columnspan=5, padx=5, 
                                          pady=5) 
        
        # show toolbar (needs own frame because calls "pack()" internallly)
        self.toolbar_frame = tk.Frame(master=self)
        self.toolbar_frame.grid(column=2, row=5, columnspan=5,
                                sticky=tk.E)
        self.toolbar = Toolbar(self.canvas, self.toolbar_frame).update() #NavigationToolbar2Tk
    
    def instructions(self):
        #create pop-up window with instruction
        top=tk.Toplevel(self)
        top.geometry("310x730")
        top.title("Instructions") 
        tk.Label(top, text=text_instructions, wraplength=300, anchor="e", justify="left").pack()
        
    def import_blank(self):
        #select blank file for analysis
        self.blank_file_dir= fd.askopenfilename(title="Select blank file", filetypes = (("Text files", "*.csv"),("All files","*.*")))
        self.df_blank = pd.read_csv(self.blank_file_dir, skiprows=3)
        self.df_blank_power = pd.read_csv(self.blank_file_dir,
                                                        skiprows=lambda x:x not in [0,1])
        self.blank_name = os.path.basename(self.blank_file_dir)
        self.label_blank.config(text=self.blank_name)

    def import_sample(self):
    #select sample file for analysis
       self.sample_file_dir= fd.askopenfilename(title="Select sample file", filetypes = (("Text files", "*.csv"),("All files","*.*")))
       self.df_sample = pd.read_csv(self.sample_file_dir, skiprows=3)
       self.df_sample_power = pd.read_csv(self.sample_file_dir,
                                                        skiprows=lambda x:x not in [0,1])
       self.sample_name = os.path.basename(self.sample_file_dir)
       self.label_sample.config(text=self.sample_name)
       
    def counts_region(self, df, df_power, lbound_low, lbound_high, fbound_low, fbound_high):
        """Add up (integrate) counts for the laser bounds (lbound) and for the 
        fluorescent bounds (fbound) based on the wavelength bounds that are
        provided.
        
        First, get wavelength range for bounds. Then, for each point in that range,
        multiply the intensity/nm of that point with the spacing of each data point (gradient).
        This converts intensity/nm to intensity. Then, sum up the intensity values. 
        Finally, normalize (by dividing) the intensity to the power measured by the Si detector (df_power).
        
        df: dataframe with spectrum intensities
        df_power: dataframe with intensities from silicon detector/multimeter
        
        Returns: df_counts which has counts from laser and fluoresent regions
        """
        laser_power_array = df_power.loc[0].array #convert dataframe into array
        
        laser_signal_boolean = (lbound_low < df["wavelength"]) & (df["wavelength"]< lbound_high)
        laser_signal = df[laser_signal_boolean]
        
        laser_signal_delta_x = np.gradient(laser_signal["wavelength"]) #get wavelength spacing
        laser_signal = laser_signal.mul(laser_signal_delta_x, axis=0) #multiply counts by wavelength spacing
        
        laser_signal_sum = laser_signal.sum() #sum all the counts
        laser_signal_sum = laser_signal_sum.drop("wavelength")
        laser_signal_norm = laser_signal_sum/laser_power_array #normalize by power
        
           
        fluor_signal_boolean = (fbound_low < df["wavelength"]) & (df["wavelength"]< fbound_high)
        fluor_signal = df[fluor_signal_boolean]
        
        fluor_signal_delta_x = np.gradient(fluor_signal["wavelength"]) #get wavelength spacing
        fluor_signal = fluor_signal.mul(fluor_signal_delta_x, axis=0) #multiply counts by wavelength spacing
        
        fluor_signal_sum = fluor_signal.sum()
        fluor_signal_sum = fluor_signal_sum.drop("wavelength")
        fluor_signal_norm = fluor_signal_sum/laser_power_array
            
        df_counts = pd.concat([laser_signal_norm, fluor_signal_norm], axis=1)
        df_counts.columns = ["excit_total_counts", "fluor_total_counts"]
    
        return df_counts
    
    def calculate_plqy(self):
        """PLQY calculation (uses "counts_region"")
        
        """
        lbound_low = int(self.entry_laser_low.get())
        lbound_high = int(self.entry_laser_high.get())
        fbound_low = int(self.entry_pl_low.get())
        fbound_high = int(self.entry_pl_high.get())
        
        self.bounds = (lbound_low, lbound_high, fbound_low, fbound_high)
        
        self.df_blank_analysis = self.counts_region(self.df_blank, self.df_blank_power,
                                                    lbound_low, 
                                                    lbound_high, fbound_low,
                                                    fbound_high)
        self.df_sample_analysis = self.counts_region(self.df_sample, self.df_sample_power,
                                                     lbound_low, 
                                                    lbound_high, fbound_low,
                                                    fbound_high)
            
        self.photons_emitted_mean = self.df_sample_analysis["fluor_total_counts"]["mean_photon_counts"] -\
                                    self.df_blank_analysis["fluor_total_counts"]["mean_photon_counts"]
        self.photons_abs_mean = self.df_blank_analysis["excit_total_counts"]["mean_photon_counts"] -\
                                self.df_sample_analysis["excit_total_counts"]["mean_photon_counts"]
        self.plqy_mean = self.photons_emitted_mean/self.photons_abs_mean
        
        self.calculate_plqy_std() #call function to calculate standard deviation
        
        self.update_plot()
        
    def calculate_plqy_std(self):
        """Error bars for PLQY calculation. 
        
        Only considers random error and not systemic error. Systemic error is probably
        about 5%
        """
        #   Averages
        blank_laser_average = self.df_blank_analysis["excit_total_counts"]["mean_photon_counts"]
        blank_fluor_average = self.df_blank_analysis["fluor_total_counts"]["mean_photon_counts"]
        sample_laser_average = self.df_sample_analysis["excit_total_counts"]["mean_photon_counts"]
        sample_fluor_average = self.df_sample_analysis["fluor_total_counts"]["mean_photon_counts"]   
        
        # std deviations
        blank_laser_std = self.df_blank_analysis["excit_total_counts"].drop("mean_photon_counts").std()
        blank_fluor_std = self.df_blank_analysis["fluor_total_counts"].drop("mean_photon_counts").std()
        sample_laser_std = self.df_sample_analysis["excit_total_counts"].drop("mean_photon_counts").std()
        sample_fluor_std = self.df_sample_analysis["fluor_total_counts"].drop("mean_photon_counts").std()
        
        #variances
        blank_laser_var = blank_laser_std**2
        blank_fluor_var = blank_fluor_std**2
        sample_laser_var = sample_laser_std**2
        sample_fluor_var =  sample_fluor_std **2
        
        # Quantum yield average and standard deviation
        qy_average = ((sample_fluor_average-blank_fluor_average)
                     /(blank_laser_average-sample_laser_average))
        qy_std_a = ((sample_fluor_var + blank_fluor_var)
                    /((sample_fluor_average-blank_fluor_average))**2)
        qy_std_b = ((blank_laser_var + sample_laser_var)
                    /(blank_laser_average-sample_laser_average)**2)
        self.plqy_std = qy_average*((qy_std_a + qy_std_b)**(1/2))

      
    def update_plot(self):
        #update plot after PLQY calculation
        self.fig.clf()
        self.fig = Figure(figsize=(6,4), dpi=100, tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_default_filename = lambda: (
            self.sample_file_dir[0:-4] + '.png') 
        self.plot_analysis = self.fig.add_subplot(111)
        
        #update toolbar
        self.toolbar_frame = tk.Frame(master=self)
        self.toolbar_frame.grid(column=2, row=5, columnspan=4,
                                sticky=tk.E)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame).update()
        
        lbound_low = int(self.entry_laser_low.get())
        lbound_high = int(self.entry_laser_high.get())
        fbound_low = int(self.entry_pl_low.get())
        fbound_high = int(self.entry_pl_high.get())
    
        ax = self.plot_analysis
        ax.clear()
        ax.plot(self.df_blank["wavelength"], self.df_blank["mean_photon_counts"], color="black")
        ax.plot(self.df_sample["wavelength"], self.df_sample["mean_photon_counts"], color="red")

        ax.set_title("\n".join(wrap(self.sample_name,60)), fontsize="10")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (counts)")
        ax.set_xlim (348, 738)
        
        axins2 = inset_axes(ax, width="100%", height="100%", loc='upper left',
                bbox_to_anchor=(0.25,0.55,0.3,0.4), bbox_transform=ax.transAxes)
        axins2.plot(self.df_blank["wavelength"], self.df_blank["mean_photon_counts"], color="black")
        axins2.plot(self.df_sample["wavelength"], self.df_sample["mean_photon_counts"], color="red")
        axins2.set_xlim(lbound_low,lbound_high)
        axins2.set_title(str(lbound_low)+" < Excit. (nm) < "+str(lbound_high), fontsize=8, pad=3)
        axins2.tick_params(axis='both', which='major', labelsize=8)
        
        axins3 = inset_axes(ax, width="100%", height="100%", loc='upper left',
                bbox_to_anchor=(0.65,0.55,0.3,0.4), bbox_transform=ax.transAxes)
        axins3.plot(self.df_blank["wavelength"], self.df_blank["mean_photon_counts"], color="black")
        axins3.plot(self.df_sample["wavelength"], self.df_sample["mean_photon_counts"], color="red")
        axins3.set_xlim(fbound_low,fbound_high)
        axins3.set_title(str(fbound_low)+" < PL (nm) < "+str(fbound_high), fontsize=8, pad=3)
        axins3.tick_params(axis='both', which='major', labelsize=8)
        
        #plot and scale dyamically
        index_low = np.abs(self.df_sample["wavelength"] - fbound_low).argmin()
        index_high = np.abs(self.df_sample["wavelength"] - fbound_high).argmin()
        
        ymin_PL = min(self.df_blank["mean_photon_counts"][index_low:index_high])
        ymax_PL = max(self.df_sample["mean_photon_counts"][index_low:index_high])
        axins3.set_ylim(ymin_PL, ymax_PL)
        
        
        #text for PLQY
        ax.text(0.3, 0.20, r"Quantum Yield ($\pm \sigma$): " + str(round(self.plqy_mean*100,1)) +
                r" $\pm$ " + str(round(self.plqy_std*100,1)) + "%" + 
                "\n\n" +
                "Photons absorbed (a.u.): " + str(int(round(self.photons_abs_mean))) +
                "\n" +
                "Photons emitted (a.u.): " + str(int(round(self.photons_emitted_mean))) 
          , transform=ax.transAxes
          ,bbox=dict(boxstyle="round",facecolor='white', alpha=0.1))
        
        
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=4,
                                          columnspan=5, padx=5, 
                                          pady=5) #side=tk.TOP)

    def save_analysis(self):
        #save analysis when button pressed 
        current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_filename = self.sample_name[:-4] + "_analysis_" + current_datetime + ".csv"
        directory = os.path.dirname(self.sample_file_dir) + "/" + self.sample_name[:-4] + "_analysis"
        name_dir = directory + "/" + save_filename
        os.makedirs(directory, exist_ok=True)
        
        #save analysis file
        with open(name_dir, 'a', newline="") as f:
            f.write("Bounds: \n")
            f.write("Excit_lower_bound, Excit_upper_bound, PL_lower_bound, PL_upper_bound \n " 
                    + str(self.bounds)[1:-1])
            f.write("\n\n")
            
            f.write("Sample: ")
            f.write(self.sample_name[:-4] + "\n")
            self.df_sample_analysis.to_csv(f)
            f.write("\n")
            
            f.write("Blank: ")
            f.write(self.blank_name[:-4] + "\n")
            self.df_blank_analysis.to_csv(f)
            f.write("\n")
            
            f.write("Photons_emitted_mean: ," + str(self.photons_emitted_mean) + '\n')
            f.write("Photons_absorbed_mean: ," + str(self.photons_abs_mean) + '\n')
            f.write("PLQY: ," + str(self.plqy_mean) + '\n')
        
        #save figure
        self.fig.savefig(name_dir[:-4] + ".png")
        initialize_frame.label_collection_progress.config(text="Analysis files saved.")
            
            
class App(tk.Tk): 
    # root window
    def __init__(self):
        super().__init__()
        
        # configure root window
        self.title("QEPro Controller and PLQY calculator")
        self.geometry("1080x710")
        
        # configure columns
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=8)
        
        self.protocol('WM_DELETE_WINDOW', quit) #modify what quit button does

def quit():
    """Modify quit button
    
    To allow instruments to disconnect safely, turn off live view and wait.
    Then close spectrometer and multimeter before quitting.
    """
    try:
        initialize_frame.live_view_bool = False
        time.sleep(0.5)
        initialize_frame.spec.close()
        initialize_frame.multimeter.shutdown()
        app.quit()
        app.destroy()
    except:
        app.quit()
        app.destroy()
        
     
class Toolbar(NavigationToolbar2Tk):
    #customize toolbar to not show coordinates which causes resizing
    def set_message(self, s):
        pass
    
if __name__ == "__main__":
    """
    Make instance of root window (app). Then, make instance of all frames
    that is placed in the root window. Then, run the the GUI using
    "app.mainloop()"
    """
    app = App()
    initialize_frame = InitializeFrame(app)
    collect_data_frame = CollectDataFrame(app)
    analyze_data_frame = AnalyzeDataFrame(app)
    app.mainloop()