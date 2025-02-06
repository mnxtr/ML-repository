import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from scipy.signal import butter , lfilter 
from sklearn.model_selection import train_test_split, Kfold 
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, Dataloader 
def load_ecg_data(file_paths, labels):
    """
    file_paths: list of file paths (e.g., one per ECG segment or record)
    labels: list or array of arrhythmia labels corresponding to file_paths
    
    return:
        ecg_signals: a list or array of raw ECG signals
        labels: same shape as input labels
    """
    ecg_signals = []
    # Example reading logic; adapt for your data format
    for fp in file_paths:
        data = pd.read_csv(fp)
        ecg_signals.append(data['ecg_values'].values)
    return ecg_signals, labels 

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(ecg_signal, lowcut=0.5, highcut=45.0, fs=360):
    b, a = butter_bandpass(lowcut, highcut, fs, order=4)
    filtered_signal = lfilter(b, a, ecg_signal)
    return filtered_signal

def preprocess_ecg(ecg_signals):
    """
    Apply bandpass filtering, normalization, or other preprocessing steps.
    """
    processed_signals = []
    for signal in ecg_signals:
        filtered = bandpass_filter(signal)
        # Example: Standard normalization
        filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)
        processed_signals.append(filtered)
    return processed_signals
