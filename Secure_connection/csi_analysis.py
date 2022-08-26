from copyreg import remove_extension
from email import header
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os

from CSIKit.util import byteops
from CSIKit.reader import NEXBeamformReader
from CSIKit.tools.batch_graph import BatchGraph
from CSIKit.util.filters import running_mean
from CSIKit.util import csitools
from sklearn import preprocessing
from sklearn.svm import OneClassSVM

def remove_fill(x,frame_no):
    # x_new = x.copy()
    x_new = pd.Series(x)
    for j in frame_no:
        value = x_new.values[j]
        x_new = x_new.replace(value,np.nan)
        x_new = x_new.interpolate()
        x_new_1 = x_new.to_numpy()
        x_new_1 = x_new_1.transpose()
    return x_new_1

def remove_nan(csi_matrix):
    nan = np.where(np.isnan(csi_matrix))
    nan_line = nan[0]
    nan_list = nan_line.tolist()
    result = [] 

    for i in nan_list: 
        if i not in result: 
            result.append(i) 

    for i in result:
        csi_matrix[i,:] = csi_matrix[i+1,:]

    return csi_matrix

def var_cal(df):

    # df = df.drop('index', axis=1)
    df = df.values.tolist()
    df = np.array(df)
    var_m = np.corrcoef(df)
    var = sum(map(sum, var_m))/(np.shape(var_m)[0] ** 2)
    
    return var



def csi_analyzer(csi_file_name):

    my_reader = NEXBeamformReader()
    # csi_file = 'iphone12pro_lab_s1.pcap'
    csi_data = my_reader.read_file("/Users/liangxintai/Desktop/live_pcap/" + csi_file_name)

    csi = csi_data.frames[0].csi_matrix   # Remove frame from data
    sc_count = np.shape(csi)[0]
    sc_idx = np.arange(sc_count) - int(sc_count / 2)     # generate index for subcarriers
    sc_count_new = sc_count - 36
    sc_idx_new = np.arange(sc_count_new) - int(sc_count_new/2)

    scidx_80mhz_csi_no_dc = [-1,0,1]
    no_dc = [x+128 for x in scidx_80mhz_csi_no_dc]
    scidx_80mhz_csi_no_pilot = [-103, -75, -39, -11, 11, 39, 75, 103]
    no_pilot = [x+128 for x in scidx_80mhz_csi_no_pilot]
    # gen_scidx_data_80mhz_csi = np.in1d(sc_idx, scidx_80mhz_csi_no_pilot)

    csi_amplitude, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude")
    csi_amplitude_subch = csi_amplitude[:,:,0,0]
    # csi_amplitude_subch = csi_amplitude_subch[:,12:244]
    # csi_amplitude_subch_buff = np.empty((no_frames, no_subcarriers))
    csi_amplitude_buff = np.zeros(np.shape(csi_amplitude_subch))

    for i in range(no_frames):
        csi_amplitude_buff[i,:] = remove_fill(csi_amplitude_subch[i,:],no_pilot)
        for j in range(6):
            # csi_phase_subch[i,:] = remove_fill(csi_phase_subch[i,:],no_pilot)
            csi_amplitude_buff[i,:] = remove_fill(csi_amplitude_buff[i,:],no_dc)
        csi_amplitude_buff[i,:] = running_mean(csi_amplitude_buff[i,:],10)

    csi_amplitude_freq = csi_amplitude_buff[:,12:244].copy()

    scoretable_freq = np.empty((no_frames,1))
    scoretable_freq_std = np.empty((no_frames,1))
    csi_amplitude_grad = np.empty((no_frames,220))

    for i in range(no_frames):
        csi_amplitude_grad[i,:] = np.gradient(csi_amplitude_buff[i,17:237])

    scoretable_amplitude_grad_std = np.zeros((no_frames,1))

    for i in range(no_frames):
        std_score = 0
        std_score = np.std(csi_amplitude_grad[i,:])
        scoretable_amplitude_grad_std[i,:] = std_score

    scoretable_amplitude_grad_std_sorted = np.argsort(scoretable_amplitude_grad_std, axis=0)
    perc = 0.8
    frame_amplitude_std_selected = scoretable_amplitude_grad_std_sorted[0:int(perc*no_frames),:]

    csi_amplitude_freq_normalized = np.empty((no_frames,np.shape(csi_amplitude_freq)[1]))
    csi_amplitude_freq = remove_nan(csi_amplitude_freq)
    csi_amplitude_freq = remove_nan(csi_amplitude_freq)

    for i in range(no_frames):
        csi_amplitude_freq_normalized[i,:] = preprocessing.normalize([csi_amplitude_freq[i,:]],norm='max')

    csi_amplitude_freq_output = []
    for i in range(no_frames):
        # if i in frame_selected and i in frame_std_selected:
        if i in frame_amplitude_std_selected:
            csi_amplitude_freq_output.append(csi_amplitude_freq_normalized[i,:])
    csi_amplitude_freq_output = np.array(csi_amplitude_freq_output)
    index1 = ['0' for _ in range(np.shape(csi_amplitude_freq_output)[0])]

    # # User profile reading
    # input_path = '/Users/liangxintai/Desktop/CSI_authentication/lab_s1/data_sample/final_User_profile_amplitude/'
    # train_file_name = 'User0.csv'

    # df = pd.read_csv(input_path + train_file_name, index_col=0, header=None)
    # df.reset_index(inplace=True)
    # df = df.reindex(np.random.permutation(df.index))

    df = pd.DataFrame(csi_amplitude_freq_output, index=index1)
    # df.reset_index(inplace=True)
    # df = df.reindex(np.random.permutation(df.index))
    # print('yes')
    return df, csi_amplitude_freq_output

def train_process(df_train):

    df_train.reset_index(inplace=True)
    df_train = df_train.reindex(np.random.permutation(df_train.index))
    x = df_train.iloc[:,1:]
    train_size = int(len(x)*0.8)

    x_train = df_train.iloc[:train_size,1:]
    return x_train

def test_process(df_test):

    df_test.reset_index(inplace=True)
    df_test = df_test.reindex(np.random.permutation(df_test.index))
    test_size = 100
    x_test = df_test.iloc[:test_size,1:]
    x_test = np.array(x_test)

    return x_test


def gen_authenticator(x_train):
    
    ocsvm = OneClassSVM(kernel='rbf',gamma='scale', nu=0.01)
    ocsvm.fit(x_train)

    return ocsvm

def authenticate(ocsvm,x_test_single):

    mid = x_test_single.reshape(1, -1)
    out_test = ocsvm.predict(mid)
    out_test = out_test[0]

    if out_test == 1:
        return b'1'
    elif out_test == -1:
        return b'0'



        
