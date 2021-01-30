import pyshark
import struct
import binascii
import sys
import os.path
from os import path
import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.ticker as ticker

import optparse
import time

import numpy as np
from numpy.random import seed

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model, load_model
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import Callback

import seaborn as sns


T = 0.00002014754



losses = []

def handleLoss(loss):
    global losses
    losses += [loss]
    print(loss)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        handleLoss(logs.get('loss'))








def autoEncoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(192, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(96, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(96, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(192, activation='relu', return_sequences=True)(L4)
    output=TimeDistributed(Dense(X.shape[2]))(L5)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    return model




def FFTFreq(N, T):
    xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    return xf.astype(int)

def convertFFT(data_set):

    N = data_set.shape[0]

    try:
        D = data_set.shape[1]
   
        data_fft = np.empty((int(N/2),1))

        for i in range(D):
            fft = np.fft.fft(data_set[:,i])
  
            fft = (2.0/N )* np.abs(fft[:N//2])

            data_fft = np.append(data_fft, fft.reshape(int(N/2), 1), axis=1)

        data_fft = np.delete(data_fft, 0, 1)  
    except:

        data_fft = np.fft.fft(data_set)
        data_fft = (2.0/N )* np.abs(data_fft[:N//2])

    return data_fft[1:] #clip 1st index because index 0 is not a valid frequency




def normalize(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    return scaler.fit_transform(data)




def windowFFT(data_set, seg_size):
    N = data_set.shape[0]
    D = data_set.shape[1]
    segs = int(N / seg_size)
    data_set = data_set[:seg_size*segs]
    #print("before")
    #print(data_set)
    data_set.shape = (segs, seg_size, D)
    #print("after", data_set)
    fft_windows = np.empty((segs, int(seg_size/2) -1, D))
    
    for i in range(segs):
        fft_windows[i] = convertFFT(data_set[i])

    
    return fft_windows


def importCSV(filename, sep=','):
    data_set = pd.read_csv(filename, sep=sep)
    if data_set.shape[1] == 5:
        data_set.columns = ['ID', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']
        data_set.drop(['ID'], axis=1, inplace=True)
    else:
        data_set.columns = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']
    return data_set

def plot4Channels(data):
    fig, ax = plt.subplots(num=None, figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
    ax.plot(data[:,0], '-', color='blue', linewidth=.5)
    ax.plot(data[:,1], '-', color='red', linewidth=.5)
    ax.plot(data[:,2], '-', color='green', linewidth=.5)
    ax.plot(data[:,3], '-', color='yellow', linewidth=.5)
    plt.show()


def avgFFTWindowsFreqs(data, batch):
    avg = np.zeros((1, data.shape[1]))

    for i in range(int(data.shape[0] / batch)):
        avg = np.append(avg, np.mean(data[batch*i:batch*(i+1)], axis=0).reshape(1,data.shape[1]), axis=0)

    return avg[1:]





def preprocess(data, fft_window_size, freq_batch_size, scaler):
    fft_max_index = int(fft_window_size / 2) - 1
    dim = data.shape[1]

    data_fft = windowFFT(data, fft_window_size)
    data_avg = np.empty((data_fft.shape[0], int(fft_max_index / freq_batch_size) * dim))

    for i in range(data_fft.shape[0]):
        data_avg[i] = avgFFTWindowsFreqs(data_fft[i], freq_batch_size).flatten()

    if scaler == "create":
        scaler = MinMaxScaler(feature_range=(0,1))
        data_scaled = scaler.fit_transform(data_avg)

        return data_scaled.reshape(data_scaled.shape[0], 1, data_scaled.shape[1]), scaler
    else:
        data_scaled = scaler.transform(data_avg)
        return data_scaled.reshape(data_scaled.shape[0], 1, data_scaled.shape[1])



def scorePredict(model, data):
    output = model.predict(data)
    output.shape = (data.shape[0], data.shape[2])
    scored = np.mean(np.abs(np.subtract(output, data.reshape(data.shape[0], data.shape[2]))), axis=1)
    return scored


def exportCsv(input, output):
    print("Writing to {}".format(output))
    with open(output, 'w') as f:
        writer = csv.writer(f)
        for row in input:
            writer.writerow(row)



colors="rgbcmyk"

dir_path = path.dirname(path.realpath(__file__))



#FILE_PATH = dir_path + "/data/converted-csv-from-capture-tool/coupon5_open-close.csv"
strike_fp = dir_path + "/data/converted-csv-from-capture-tool/strike-s4_.csv"
pump_only_fp = dir_path + "/data/converted-csv-from-capture-tool/pump_only.csv"
#FILE_PATH = dir_path + "/data/converted-csv-from-capture-tool/hole open c 5.raw.csv"
#FILE_PATH = dir_path + "/data/converted-csv-from-capture-tool/leak side.raw.csv"
#FILE_PATH = dir_path + "/data/converted-csv-from-capture-tool/top_hole.csv"
leak_data_fp = dir_path + "/data/pcap-to-csv/leak_large_vertical_section.csv"
little_leak_data_fp = dir_path + "/data/pcap-to-csv/leak_vertical_small.csv"
c3_leak_fp = dir_path + "/data/pcap-to-csv/coupon_3_hole_unplugged.csv"
c4_leak_fp = dir_path + "/data/pcap-to-csv/coupon_4_hole_unplugged.csv"
pump_turning_on = dir_path + "/data/pcap-to-csv/pump_turning_on.csv"
normal_data_fp = dir_path + "/data/pcap-to-csv/pump_active_normal.csv"
normal_data_2_fp = dir_path + "/data/pcap-to-csv/pump_active_normal_2.csv"
bypass_valve_fp = dir_path + "/data/pcap-to-csv/valve_turned.csv"
main_valve_fp = dir_path + "/data/pcap-to-csv/valve_partially_closed.csv"
tank_empty_fp = dir_path + "/data/pcap-to-csv/tank_emptying.csv"
tape_leak_data_fp= dir_path + "/data/pcap-to-csv/leak_tape_weak_vertical.csv"



FFT_WINDOW_SIZE = 10000
FFT_MAX_INDEX = int(FFT_WINDOW_SIZE / 2) - 1
FREQ_BATCH_SIZE = 500
DIM = 4
EPOCHS = 50


print("Importing files...")
normal_data = np.array(importCSV(normal_data_fp))
normal_data_2 = np.array(importCSV(normal_data_2_fp))
pump_only = np.array(importCSV(pump_only_fp))
tank_emptying = np.array(importCSV(tank_empty_fp))
little_leak = np.array(importCSV(little_leak_data_fp))
strike = np.array(importCSV(strike_fp, sep='\t'))
leak_data = np.array(importCSV(leak_data_fp))
bypass_valve = np.array(importCSV(bypass_valve_fp))
main_valve = np.array(importCSV(main_valve_fp))
tape_leak_data = np.array(importCSV(tape_leak_data_fp))
c3_leak = np.array(importCSV(c3_leak_fp))
c4_leak = np.array(importCSV(c4_leak_fp))


normal_data_set = {
"Normal Active Pump 1" : normal_data,
"Normal Active Pump 2" : pump_only,
"Main Valve" : main_valve[:500000],
"Tank Emptying": tank_emptying[:5000000],
"1/8in Leak" : little_leak[:500000],
#"Strikes" : strike[200000:],
"1/4in Leak" : leak_data[:700000],
"Bypass Valve Open" : bypass_valve[:1500000],
"Normal Active Pump 3" : normal_data_2
}



test_data_set = {
    "Tank Emptying" : tank_emptying,
    "1/8in Leak" : little_leak,
    "Pipeline Impact" : strike,
    "1/4in Leak" : leak_data,
    "Bypass Valve Open" : bypass_valve,
    "Main Valve Partial Close" : main_valve,
    "Leaking Through Tape" : tape_leak_data,
    "Coupon 4 Opened" :c4_leak,
    "Coupon 3 Opened" : c3_leak
}



healthy_data = normal_data

for key in normal_data_set:
    healthy_data = np.concatenate((healthy_data, normal_data_set[key]), axis=0)


train, scaler = preprocess(healthy_data, FFT_WINDOW_SIZE, FREQ_BATCH_SIZE, scaler="create")

preprocessed_test_set = test_data_set
preprocessed_train_set = normal_data_set

for key in test_data_set:
    preprocessed_test_set[key] = preprocess(test_data_set[key], FFT_WINDOW_SIZE, FREQ_BATCH_SIZE, scaler=scaler)


for key in normal_data_set:
    preprocessed_train_set[key] = preprocess(normal_data_set[key], FFT_WINDOW_SIZE, FREQ_BATCH_SIZE, scaler=scaler)




test = preprocess(tank_emptying,FFT_WINDOW_SIZE, FREQ_BATCH_SIZE, scaler=scaler)

model = autoEncoder_model(train)



history = model.fit(train, train, epochs=EPOCHS, batch_size=10, validation_split=0.05).history


fig, ax = plt.subplots(num=None, figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
ax.plot(history['loss'], label='loss mae')
ax.plot(history['val_loss'], label='validation')

ax.legend(loc='upper right')
plt.show()


model.save("FFT_LSTM_"+str(FFT_WINDOW_SIZE)+"_"+str(FREQ_BATCH_SIZE)+".h5")


scored = scorePredict(model, train)

plt.figure(figsize=(16,9), dpi=80)
plt.title("Loss Distribution of Training Set")
sns.distplot(scored, bins=20, kde=True, color='blue')
plt.xlim([0.0,0.5])



plt.show()

fig, ax = plt.subplots(num=None, figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
ax.plot(scored, label='loss mae')

plt.title("Scoring Data")

offset = 0
i = 1
for key in preprocessed_train_set:
    offset += preprocessed_train_set[key].shape[0]
    ax.axvline(x=offset,color=colors[6%i], linestyle='--',label=key)
    i += 1
plt.ylim(0,.5)
ax.legend(loc='upper right')
plt.show()


threshold = 0.075


for key in preprocessed_test_set:

    scored_test = scorePredict(model, preprocessed_test_set[key])

    scored_plot = np.concatenate((scored, scored_test), axis=0)
    threshold_arr = np.full((scored_plot.shape[0]), threshold)

    cross_threshold = np.array(np.nonzero(scored_plot > threshold)).flatten()

    marks = cross_threshold
    
    last = 0
    #for val in cross_threshold:
    #    if last + 1 < val:
    #        marks.append(int(val))
    #    last=val

    fig, ax = plt.subplots(num=None, figsize=(16, 9), dpi=200, facecolor='w', edgecolor='k')

    for m in marks:
        ax.axvline(x=m,color='r',linestyle='--')
    
    ax.plot(scored_plot, label='loss mae')
    ax.legend(loc='upper right')
    plt.ylim(0,.5)
    plt.title(key)
    ax.axvline(x=train.shape[0],color='k',linestyle='--',label='End Training Data')



    plt.show()

exit(0)



