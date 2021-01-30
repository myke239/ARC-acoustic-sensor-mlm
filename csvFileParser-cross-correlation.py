import pyshark
import struct
import binascii
import sys
import os.path
from os import path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.ticker as ticker

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
        ax.plot(data_set[data_set.columns[3-i]], label=data_set.columns[3-i], linewidth=1)
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

T = 0.00002014754

dir_path = path.dirname(path.realpath(__file__))

#FILE_PATH = dir_path + "data/converted-csv-from-capture-tool/coupon5_open-close.csv"
#FILE_PATH = dir_path + "data/converted-csv-from-capture-tool/strike-s4_.csv"
#FILE_PATH = dir_path + "data/converted-csv-from-capture-tool/pump_only.csv"
#FILE_PATH = dir_path + "data/converted-csv-from-capture-tool/hole open c 5.raw.csv"
#FILE_PATH = dir_path + "data/converted-csv-from-capture-tool/leak side.raw.csv"
#FILE_PATH = dir_path + "data/converted-csv-from-capture-tool/top_hole.csv"
FILE_PATH = dir_path + "data/pcap-to-csv/leak_large_vertical_section.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/leak_vertical_small.csv"

TS_SAMPLE = 60
EXCEL_ROWS = 5
NUM_ACTIVE_CHANNELS = 4


print("Reading CSV File")
data_set = pd.read_csv(FILE_PATH, sep=',')
print("Data Set Shape:", data_set.shape)
data_set.columns = ['ID', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']
data_set.drop(['ID'], axis=1, inplace=True)

plotChannels(data_set, 4)

windowSize = 2000

segment = data_set[2000000:2020000]
segment.columns = data_set.columns

plotChannels(segment, 4)


timestamps = np.arange(start=-int(windowSize / 2), stop=int(windowSize/2), step=1, dtype=float)
timestamps *=  T*1000

c_vals = crossCorrFunction(segment['Channel 1'], segment['Channel 3'], windowSize)

offset_index = c_vals.index(max(c_vals))
print(max(c_vals))
print(offset_index)

time_offset = ((-windowSize/2) + offset_index)*T*1000

fig, ax = plt.subplots(figsize=(14,6), dpi=80)



ax.xaxis.set_major_locator(ticker.AutoLocator())
ax.xaxis.set_minor_locator(ticker.AutoLocator())

ax.axvline(x=0,color='k',linestyle='--',label='Center')
ax.axvline(x=time_offset,color='r',linestyle='--',label='Peak Synchrony = {:.2f}ms'.format(time_offset), linewidth=1)

ax.plot(timestamps, c_vals, linewidth=2)

plt.xlabel("Time Lag (ms)")
plt.ylabel("Correlation Coefficient")

plt.legend(loc='lower left')
ax.set_title('Time lag between Ch1 and Ch3 determined by Cross-Correlation', fontsize=16)
plt.show()