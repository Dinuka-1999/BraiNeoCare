import mne
import os
import numpy as np
import scipy
import spkit as sp
import matplotlib.pyplot as plt

# Load the annotation file. Replace the file path with the path to the annotation file.
Annotations=scipy.io.loadmat('../BraiNeoCare/Datasets/zenodo_eeg/annotations_2017.mat')

# Defined function to calculate the signal array
# For 12 channels the signal array is calculated as follows:
def signal_array_12(raw_data):
    ch1=raw_data[0]-raw_data[5] #FP1-T3
    ch2=raw_data[5]-raw_data[7] #T3-O1
    ch3=raw_data[0]-raw_data[2] #FP1-C3
    ch4=raw_data[2]-raw_data[7] #C3-O1
    ch5=raw_data[1]-raw_data[3] #FP2-C4
    ch6=raw_data[3]-raw_data[8] #C4-O2
    ch7=raw_data[1]-raw_data[6] #FP2-T4
    ch8=raw_data[6]-raw_data[8] #T4-O2
    ch9=raw_data[5]-raw_data[2] #T3-C3
    ch10=raw_data[2]-raw_data[4] #C3-Cz
    ch11=raw_data[4]-raw_data[3] #Cz-C4
    ch12=raw_data[3]-raw_data[6] #C4-T4
    
    return np.array([ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12])

# For 18 channels the signal array is calculated as follows:
def signal_array_18(raw_data):
    ch1=raw_data[1]-raw_data[3] #FP2-F4
    ch2=raw_data[3]-raw_data[5] #F4-C4
    ch3=raw_data[5]-raw_data[7] #C4-P4
    ch4=raw_data[7]-raw_data[9] #P4-O2
    ch5=raw_data[0]-raw_data[2] #FP1-F3
    ch6=raw_data[2]-raw_data[4] #F3-C3
    ch7=raw_data[4]-raw_data[6] #C3-P3
    ch8=raw_data[6]-raw_data[8] #P3-O1
    ch9=raw_data[1]-raw_data[11] #FP2-F8
    ch10=raw_data[11]-raw_data[13] #F8-T4
    ch11=raw_data[13]-raw_data[15] #T4-T6
    ch12=raw_data[15]-raw_data[9] #T6-O2
    ch13=raw_data[0]-raw_data[10] #FP1-F7
    ch14=raw_data[10]-raw_data[12] #F7-T3
    ch15=raw_data[12]-raw_data[14] #T3-T5
    ch16=raw_data[14]-raw_data[8] #T5-O1
    ch17=raw_data[16]-raw_data[17] #FZ-CZ
    ch18=raw_data[17]-raw_data[18] #CZ-PZ
    
    return np.array([ch9,ch10,ch11,ch12,ch4,ch3,ch2,ch1,ch17,ch18,ch13,ch14,ch15,ch16,ch8,ch7,ch6,ch5])

# Read the annotation file and find the seizure time
def find_seizure_time(file_no):
    Annotations_by_consensus=Annotations['annotat_new'][0][file_no-1][0] & Annotations['annotat_new'][0][file_no-1][1] & Annotations['annotat_new'][0][file_no-1][2]
    a=np.where(Annotations_by_consensus==1)[0]
    start_time=[]
    end_time=[]
    if len(a)!=0:
        start_time.append(a[0])  
        for r in range(1,a.shape[0]):
            if a[r-1]-a[r]!=-1:
                end_time.append(a[r-1]+1)
                start_time.append(a[r]) 
        end_time.append(a[-1]+1)  
        
    return np.array(start_time),np.array(end_time)

# Signal filtering and downsampling
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

def check_consecutive_zeros(arr):
    return np.all(arr==0)

def removeArtifacts(signal1):
    beta_val=0.3
    signal1=signal1*3000000
    signal1=bandpassFilter(signal1)
    signal1=scipy.signal.resample_poly(signal1,up=128,down=256,axis=1)
    clean_array= sp.eeg.ATAR(signal1.T, wv='db4', winsize=128, beta=beta_val, thr_method='ipr', OptMode='soft',verbose=0)
    clean_array=clean_array.T/3000000
    signal1=scipy.signal.resample_poly(clean_array,up=32,down=128,axis=1)
    return signal1

channels_1_18=['EEG Fp1-Ref', 'EEG Fp2-Ref','EEG F3-Ref', 'EEG F4-Ref', 'EEG C3-Ref', 'EEG C4-Ref','EEG P3-Ref',
            'EEG P4-Ref','EEG O1-Ref', 'EEG O2-Ref','EEG F7-Ref','EEG F8-Ref', 'EEG T3-Ref', 'EEG T4-Ref','EEG T5-Ref', 'EEG T6-Ref','EEG Fz-Ref','EEG Cz-Ref','EEG Pz-Ref' ]
channels_2_18=['EEG Fp1-REF', 'EEG Fp2-REF','EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF','EEG P3-REF',
            'EEG P4-REF','EEG O1-REF', 'EEG O2-REF','EEG F7-REF','EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF','EEG T5-REF', 'EEG T6-REF','EEG Fz-REF','EEG Cz-REF','EEG Pz-REF' ]

channels_1_12=['EEG Fp1-Ref', 'EEG Fp2-Ref',  'EEG C3-Ref', 'EEG C4-Ref', 'EEG Cz-Ref', 'EEG T3-Ref','EEG T4-Ref', 'EEG O1-Ref', 'EEG O2-Ref']
channels_2_12=['EEG Fp1-REF', 'EEG Fp2-REF',  'EEG C3-REF', 'EEG C4-REF', 'EEG Cz-REF', 'EEG T3-REF','EEG T4-REF', 'EEG O1-REF', 'EEG O2-REF']

channels={"121":channels_1_12,"122":channels_2_12,"181":channels_1_18,"182":channels_2_18}

def read_file(file,n,plot_=False,number_of_channel=12,pre_processing=True):
    data = mne.io.read_raw_edf(file)
    try:
        data=data.pick_channels(channels[f"{number_of_channel}1"],ordered=True)
        raw_data=data.get_data()
    except:
        data=data.pick_channels(channels[f"{number_of_channel}2"],ordered=True)
        raw_data=data.get_data()

    if number_of_channel==12:
        signal=signal_array_12(raw_data)
    else:
        signal=signal_array_18(raw_data)

    s_Time,e_Time=find_seizure_time(n) 

    if pre_processing:
        signal=downsample(signal)

    if plot_:
        channel_names=["Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2","Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"]
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
    else:
        return signal,s_Time,e_Time

if __name__=="__main__":

    folder='../BraiNeoCare/Datasets/zenodo_eeg/'
    train_signals=[]
    train_seizure=[]
    test_signals=[]
    test_seizure=[]

    files=os.listdir(folder)
    np.random.shuffle(files)
    c=1
    for file in files:
        if file.endswith('.edf'):

            signal1,s_Time,e_Time=read_file(folder+file,int(file.split('.')[0][3:]),pre_processing=False)

            if (len(s_Time)!=0 ):
                signal=downsample(signal1)
                u_bound=384
                l_bound=0
                started=False

                while u_bound<signal.shape[1]:
                    
                    partition=signal[:,l_bound:u_bound]

                    if np.any((s_Time*32>=l_bound) & (s_Time*32<=u_bound-1)):
                        if 1<=c<=8:
                            test_signals.append(partition)
                            test_seizure.append(1)
                        else:
                            train_signals.append(partition)
                            train_seizure.append(1)
                        
                        started=True
                        u_bound+=32
                        l_bound+=32

                    elif np.any((l_bound<(e_Time*32-1)) & ((e_Time*32-1)<=(u_bound-1))):
                        if 1<=c<=8:
                            test_signals.append(partition)
                            test_seizure.append(1)
                        else:
                            train_signals.append(partition)
                            train_seizure.append(1)
                        started=False
                        u_bound+=32
                        l_bound+=32

                    elif (started and np.any((e_Time*32-1)>u_bound-1)):
                        if 1<=c<=8:
                            test_signals.append(partition)
                            test_seizure.append(1)
                        else:
                            train_signals.append(partition)
                            train_seizure.append(1)
                        u_bound+=32
                        l_bound+=32
                    else:
                        if check_consecutive_zeros(signal1[:,l_bound//32*256:u_bound//32*256]):
                            pass
                        else:
                            if 1<=c<=8:
                                test_signals.append(partition)
                                test_seizure.append(0)
                            else:
                                train_signals.append(partition)
                                train_seizure.append(0)
                        u_bound+=64
                        l_bound+=64
                c+=1
            

    test_signals=np.array(test_signals)
    test_seizure=np.array(test_seizure)
    train_signals=np.array(train_signals)
    train_seizure=np.array(train_seizure)
    np.save('../BraiNeoCare/Datasets/GAT/Artifacts_removed/testdata.npy',test_signals)
    np.save('../BraiNeoCare/Datasets/GAT/Artifacts_removed/testlabels.npy',test_seizure)
    np.save('../BraiNeoCare/Datasets/GAT/Artifacts_removed/traindata.npy',train_signals)
    np.save('../BraiNeoCare/Datasets/GAT/Artifacts_removed/trainlabels.npy',train_seizure)