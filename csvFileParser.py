import pyshark
import struct
import binascii
import sys
import os.path
from os import path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

import optparse
import time

import numpy as np
from numpy.random import seed

import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns




def plotChannels(data_set, channels):
    fig, ax = plt.subplots(figsize=(14,6), dpi=80)
    for i in range(channels):
        ax.plot(data_set[data_set.columns[i]], label=data_set.columns[i], linewidth=1)
    plt.legend(loc='lower left')
    ax.set_title('Data', fontsize=16)
    plt.show()

def FFTFreq(N, T):
    xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    return xf.astype(int)

def convertFFT(data_set, type, time_interval):

    N = data_set.shape[0]
    if type == "DataFrame":
        df = pd.DataFrame()
        channels = data_set.shape[1]

        xf = np.linspace(0.0, 1.0/(2.0*time_interval), int(N/2))
        
        for i in range(channels):
            data_fft = np.fft.fft(data_set[data_set.columns[i]])
            data_fft = (2.0/N )* np.abs(data_fft[:N//2])
            column_name = "Channel " + str(i)
            df.insert(i, column_name, data_fft, True)

        return df
    
    elif type == "1DNP":
        data_fft = np.fft.fft(data_set)
        data_fft = (2.0/N )* np.abs(data_fft[:N//2])

        return data_fft

    else:
        print("No comprendo!")



def normalize(capture_data):
    scaler = MinMaxScaler()
    scaler_data = scaler.fit_transform(capture_data)
    return scaler, scaler_data



def crossCorrFunction(sigA, sigB, windowSize):

    coefficient = []


    for frameLag in range(int(windowSize / 2) *(-1), int(windowSize/2)):
        coefficient.append(sigA.corr(sigB.shift(frameLag), method='pearson'))

    return coefficient

T = 0.0000208125

dir_path = path.dirname(path.realpath(__file__))

#FILE_PATH = dir_path + "data/pcap-to-csv/coupon5_open-close.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/strike-s4_.csv"
FILE_PATH = dir_path + "data/pcap-to-csv/pump_only.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/hole open c 5.raw.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/leak side.raw.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/top_hole.csv"


TS_SAMPLE = 60
EXCEL_ROWS = 5
NUM_ACTIVE_CHANNELS = 4


print("Reading CSV File")
data_set = pd.read_csv(FILE_PATH, sep=',')
print("Data Set Shape:", data_set.shape)
data_set.columns = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']


#plotChannels(data_set, 4)



windowSize = 500

scaler, scaled_data = normalize(data_set)

print(scaled_data.shape)
segment = pd.DataFrame(scaled_data)
segment.columns = data_set.columns

plotChannels(segment, 4)

c_vals = crossCorrFunction(segment['Channel 3'], segment['Channel 4'], windowSize)

fig, ax = plt.subplots(figsize=(14,6), dpi=80)

ax.plot(range(-int(windowSize / 2), int(windowSize/2)), c_vals, linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Data', fontsize=16)
plt.show()

exit()