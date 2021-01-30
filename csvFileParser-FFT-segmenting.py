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


T = 0.00002014754


def plotDataFrameChannelsRaw(data_set, title):
    
    num_x_vals = data_set.shape[0]
    total_time_elapsed = num_x_vals * T
    timestamps = np.arange(start=0, stop=num_x_vals, step=1, dtype=float)
    timestamps *= T

    fig, ax = plt.subplots(figsize=(14,6), dpi=200)
    for i in range(data_set.shape[1]):
       ax.plot(data_set[data_set.columns[3-i]], label=data_set.columns[3-i], linewidth=.1)

    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoLocator())

    plt.xlabel("Time (seconds)")
    plt.ylabel("Signal Amplitude")
    plt.legend(loc='best')
    ax.set_title(title, fontsize=16)
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






def windowFFT(data_set, seg_size, column_index):
    
    segment = data_set[:seg_size]
    segment_fft = convertFFT(np.array(segment[data_set.columns[column_index]]), "1DNP", 0).reshape(int(seg_size/2), 1)

    i = 1
    while True:

        segment = data_set[i*seg_size: (i+1)*seg_size]

        if(segment.shape[0] == seg_size):
            seg_fft = convertFFT(np.array(segment[data_set.columns[column_index]]), "1DNP", 0)
            segment_fft = np.append(segment_fft, seg_fft.reshape(int(seg_size/2), 1), axis=1)

        else:
            break

        i += 1

    print(segment_fft[0].shape)

    timestamps = np.arange(start=0, stop=segment_fft[0].shape[0], step=segment_fft[0].shape[0] / 10, dtype=float)
    timestamps *= (seg_size * T)
    return timestamps, segment_fft



def importCSV(filename):
    data_set = pd.read_csv(filename, sep=',')
    if data_set.shape[1] == 5:
        data_set.columns = ['ID', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']
        data_set.drop(['ID'], axis=1, inplace=True)
    else:
        data_set.columns = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']
    return data_set


dir_path = path.dirname(path.realpath(__file__))

#FILE_PATH = dir_path + "data/converted-csv-from-capture-tool/coupon5_open-close.csv"
#FILE_PATH = dir_path + "data/converted-csv-from-capture-tool/strike-s4_.csv"
#FILE_PATH = dir_path + "data/converted-csv-from-capture-tool/pump_only.csv"
#FILE_PATH = dir_path + "data/converted-csv-from-capture-tool/hole open c 5.raw.csv"
#FILE_PATH = dir_path + "data/converted-csv-from-capture-tool/leak side.raw.csv"
#FILE_PATH = dir_path + "data/converted-csv-from-capture-tool/top_hole.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/leak_large_vertical_section.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/leak_small_vertical.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/coupon_3_hole_unplugged.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/coupon_4_hole_unplugged.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/pump_turning_on.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/pump_active_normal.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/pump_active_normal_2.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/valve_turned.csv"
FILE_PATH = dir_path + "data/pcap-to-csv/valve_partially_closed.csv"
#FILE_PATH = dir_path + "data/pcap-to-csv/tank_emptying.csv"


data_set = importCSV(FILE_PATH)






plotDataFrameChannelsRaw(data_set, "Raw Acoustic Signals - x")


exit(0)
seg_size = 5000

timestamps, segment_fft = windowFFT(data_set, seg_size, 2)



x_time = []
for t in timestamps:
    x_time.append("{:.3f}".format(t))

fig, ax = plt.subplots(figsize=(14,6), dpi=80)
for i in range(10, int(seg_size/2)):
    ax.plot(segment_fft[i], linewidth=.25)


plt.xticks( np.arange(start=0, stop=segment_fft[0].shape[0], step=segment_fft[0].shape[0] / 10, dtype=int), labels=x_time)
plt.xlabel("Time (seconds)")
plt.ylabel("Signal Amplitude")
plt.legend(loc='lower left')
ax.set_title('Windowed FFT Plot', fontsize=16)
plt.show()


batch = 100
avg = np.zeros((1, segment_fft.shape[1]))
for i in range(int(segment_fft.shape[0] / batch)):
    avg = np.append(avg, np.mean(segment_fft[batch*i:batch*(i+1)], axis=0).reshape(1,segment_fft.shape[1]), axis=0)

print(avg.shape)


fig, ax = plt.subplots(figsize=(14,6), dpi=80)
for i in range(avg.shape[0]):
    ax.plot(avg[i], linewidth=.25)

plt.xlabel("Time (seconds)")
plt.ylabel("Signal Amplitude")
plt.legend(loc='lower left')
ax.set_title('Windowed FFT Plot (Averaged Frequencies)', fontsize=16)
ax.set_title('Data', fontsize=16)
plt.show()


