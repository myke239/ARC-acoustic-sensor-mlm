

<h3 align="center">Autoencoder LSTM Model for Anomaly Detection in Acoustic Data From Pipes</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> A machine learning model and data preprocessing scripts for detecting anomalies in pipe systems using Python, Keras, Tensorflow, and acoustic sensor data. Researched and developed for FIU Applied Research Center for Master Thesis in Electrical Engineering.
    <br> 
</p>


## 🧐 About <a name = "about"></a>



## 🏁 Getting Started <a name = "getting_started"></a>

Currently only tested on Ubuntu 20.04 with Python 3.8.5 but will most likely work on windows and other versions of python.

### Prerequisites
<br/>

The main Python and package version requirements are:
```
Python=3.8.5 and virtual environment
Tensorflow=2.2.0
Keras=2.3.1
```
<br/><br/>
Installing Python3 and dependencies.
```
sudo apt update
sudo apt install python3.8
sudo apt install python3-dev python3-pip python3-venv
sudo apt install wget
```
If you have an nVidia graphics card with CUDA support you can install the CUDA toolkit
so tensorflow can run directly on your GPU.
```
sudo apt install nvidia-cuda-toolkit
```


<br/><br/>
Check which version of Python you have installed:
```
python3
Python 3.8.5 (default, Jul 28 2020, 12:59:40) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```
If there was already a different version of Python3 installed, it's not a problem,
you'll just have to run ```python3.8``` instead of ```python3``` for commands below.
<br/><br/><br/>

### Download Code

Clone the repository into your home folder or wherever you want the project to be.
```
cd ~/
git clone https://github.com/myke239/ARC-acoustic-sensor-mlm.git
```
<br/><br/>
Change directories into folder you just downloaded.
```
cd ~/ARC-acoustic-sensor-mlm
```
<br/><br/>
Create python virtual environment and activate it.
```
~/ARC-acoustic-sensor-mlm$: python3 -m venv --system-site-packages ./venv
~/ARC-acoustic-sensor-mlm$: source ./venv/bin/activate  # sh, bash, or zsh
```
<br/><br/>
To leave the python virtual environment.
```
~/ARC-acoustic-sensor-mlmdeactivate
```
<br/><br/>
In the virtual environment install Python package requirements for the project.
```
cd ~/ARC-acoustic-sensor-mlm
~/ARC-acoustic-sensor-mlm$: source ./venv/bin/activate
(venv) ~/ARC-acoustic-sensor-mlmpip install --upgrade pip
(venv) ~/ARC-acoustic-sensor-mlmpip install -r requirements.txt
```
<br/><br/>
Verify Tensorflow was installed correctly:
```
(venv) ~/ARC-acoustic-sensor-mlmpython -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

As long as you don't have any Python module related errors you should be fine, the last part of the output should be:
```
tf.Tensor(324.51093, shape=(), dtype=float32)
```

There may be a number of messages from tensorflow about not being able to load dynamic libraries:
```
coreClock: 1.785GHz coreCount: 16 deviceMemorySize: 3.82GiB deviceMemoryBandwidth: 119.24GiB/s
2021-01-23 14:54:52.481776: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
2021-01-23 14:54:52.481883: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcublas.so.10'; dlerror: libcublas.so.10: cannot open shared object file: No such file or directory
2021-01-23 14:54:52.481980: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory
2021-01-23 14:54:52.482042: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory
2021-01-23 14:54:52.482171: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory
2021-01-23 14:54:52.482310: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10'; dlerror: libcusparse.so.10: cannot open shared object file: No such file or directory
2021-01-23 14:54:52.482485: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudnn.so.7'; dlerror: libcudnn.so.7: cannot open shared object file: No such file or directory
2021-01-23 14:54:52.482493: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1598] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
```
These warnings can be ignored for now. It means that tensorflow will run on your CPU instead of GPU. Tensorflow related tasks will take longer but these warnings won't stop the program from running.
<br/><br/><br/>
### Download Example Data
In the project folder download example acoustic data. It doesn't matter if you do this from within the virtual environment or not. (IT'S A LOT OF DATA)
```sh
cd ~/ARC-acoustic-sensor-mlm
git clone https://github.com/myke239/ARC-CEL-acoustic-sensor-data.git
```
Rename the repo folder to `data` for ease later.
```sh
mv ARC-CEL-acoustic-sensor-data data
```
<br/><br/>
Unzip the folders containing the raw acoustic sensor data.
```
cd ~/ARC-acoustic-sensor-mlm/data/wireshark/tank_emptying
7z e tank_emptying.zip.001
rm -r tank_emptying*
cd ..
unzip '*.zip'
cd ~/ARC-acoustic-sensor-mlm/data/pcap-to-csv
7z e pcap-to-csv.zip.001
rm -r pcap-to-csv*
```

## 🔧 Running the tests <a name = "tests"></a>

To run the full LSTM model and get the results in my paper:
```
cd ~/ARC-acoustic-sensor-mlm
source ./venv/bin/activate
python LSTM-Anomaly-Detection.py
```
That will spit out all the results.

There are several other python scripts that I haven't had time to document:
- csvFileParser-cross-correlation.py
- csvFileParser-FFT-segmenting.py
- csvFileParser.py
- udpFileParser.py

# 

## ✍️ Authors <a name = "authors"></a>

  - [@myke239](https://github.com/myke239) - Michael Thompson


## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- FIU College of Engineering
- FIU Applied Research Center
- Dr. Aparna Aravelli


# TODO
- Add more comments in code
- Add examples on how to use other scripts
- Add example on how to use udpFileParser.py
