import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy

mat=scipy.io.loadmat('../BraiNeoCare/Datasets/zenodo_eeg/annotations_2017.mat')

def signal_array(raw_data):
    ch1=raw_data[0]-raw_data[5]
    ch2=raw_data[5]-raw_data[7]
    ch3=raw_data[0]-raw_data[2]
    ch4=raw_data[2]-raw_data[7]
    ch5=raw_data[1]-raw_data[3]
    ch6=raw_data[3]-raw_data[8]
    ch7=raw_data[1]-raw_data[6]
    ch8=raw_data[6]-raw_data[8]
    ch9=raw_data[5]-raw_data[2]
    ch10=raw_data[2]-raw_data[4]
    ch11=raw_data[4]-raw_data[3]
    ch12=raw_data[3]-raw_data[6]
    return np.array([ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12])

def find_seizure_time(file_no):
    b=mat['annotat_new'][0][file_no-1][0] & mat['annotat_new'][0][file_no-1][1] & mat['annotat_new'][0][file_no-1][2]
    a=np.where(b==1)[0]
    s_time=[]
    e_time=[]
    if len(a)!=0:
        s_time.append(a[0])  
        for r in range(1,a.shape[0]):
            if a[r-1]-a[r]!=-1:
                e_time.append(a[r-1]+1)
                s_time.append(a[r]) 
        e_time.append(a[-1]+1)  
        
    return np.array(s_time),np.array(e_time)

fs=256
fs_d=32
order=7
resampling_factor=fs_d/fs

l_cut=30
h_cut=1

b_low,a_low=scipy.signal.cheby2(order,20,l_cut,'low',fs=fs)
b_high,a_high=scipy.signal.cheby2(order,20,h_cut,'high',fs=fs)

def bandpassFilter(signal1):
    lowpasss=scipy.signal.filtfilt(b_low,a_low,signal1,axis=1)
    return scipy.signal.filtfilt(b_high,a_high,lowpasss,axis=1)

def downsample(signal1):
    filter_signal=bandpassFilter(signal1)
    new_n_samples=int(signal1.shape[1]*resampling_factor)
    return scipy.signal.resample(filter_signal,new_n_samples,axis=1)

def read_file(file,n):

    data = mne.io.read_raw_edf(file)
    try:
        data=data.pick_channels(['EEG Fp1-Ref', 'EEG Fp2-Ref', 'EEG C3-Ref', 'EEG C4-Ref', 
                                'EEG Cz-Ref', 'EEG T3-Ref', 'EEG T4-Ref', 'EEG O1-Ref', 'EEG O2-Ref'],ordered=True)
        raw_data=data.get_data()
        signal=signal_array(raw_data)
    except:
        data=data.pick_channels(['EEG Fp1-REF', 'EEG Fp2-REF', 'EEG C3-REF', 'EEG C4-REF', 
                                'EEG Cz-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG O1-REF', 'EEG O2-REF'],ordered=True)
        raw_data=data.get_data()
        signal=signal_array(raw_data)
    channel_names=["Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2","Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"]
    signal=downsample(signal)
    s_Time,e_Time=find_seizure_time(n) 
    seizures=np.zeros((12,len(signal[0])))
    print(s_Time,e_Time)
    for i in range(12):
        for r in range(len(s_Time)):
            seizures[i,s_Time[r]*32:e_Time[r]*32]=signal[i,s_Time[r]*32:e_Time[r]*32]

    fig,ax=plt.subplots(12,1,figsize=(20,20))
    for r in range(12):
        ax[r].plot(signal[r])
        ax[r].plot(seizures[r])
        ax[r].set_title(channel_names[r])
    fig.tight_layout()
    plt.show() 

    return signal,seizures
