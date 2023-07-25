"""
PTB-XL - Lead Interpolation Initial Testing / Visualisation of Leads 

Course: Master's in Artificial Intelligence
Student: Jacob Casey
Date: March 26, 2023
"""

#Jacob Casey - 03/03/2023
import tensorflow as tf
import numpy as np
import wfdb
import matplotlib.pyplot as plt

data_directory = 'C:/Users/Jacob/University/KCL/Research Proj/data/ptb_xl_v1.0.3/records100/01000/01635_lr'

# Define the name of the ECG record to load
record_name = ''

# Load the ECG signal data and metadata
signal, metadata = wfdb.rdsamp(data_directory)

# Print the signal data shape and metadata
print('Signal data shape:', signal.shape)
print('Metadata:', metadata)

# Get the names of the leads
lead_names = metadata['sig_name']

# Create a figure with subplots for each lead
fig, axs = plt.subplots(nrows=6, ncols=2)
axs = axs.flatten()

# Plot each lead on a subplot
for i in range(signal.shape[1]):
    axs[i].plot(signal[:, i])
    axs[i].set_title(lead_names[i])
    axs[i].set_xlabel('Sample')
    axs[i].set_ylabel('Amplitude (mV)')

# Increase the spacing between subplots
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=1, hspace=1)

plt.rcParams['font.size'] = 2

# Set the DPI of the plot to 50
fig.set_dpi(100)

plt.show()
