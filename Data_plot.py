import mne
import matplotlib.pyplot as plt
import numpy as np

import scipy.io
mat=scipy.io.loadmat('../ML_codes/zenodo_eeg/annotations_2017.mat')

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

data = mne.io.read_raw_edf('../ML_codes/zenodo_eeg/eeg7.edf')
data=data.pick_channels(['EEG Fp1-Ref', 'EEG Fp2-Ref',  'EEG C3-Ref', 'EEG C4-Ref', 'EEG Cz-Ref', 'EEG T3-Ref', 'EEG T4-Ref', 'EEG O1-Ref', 'EEG O2-Ref'])
raw_data=data.get_data()
signal=signal_array(raw_data)
# for r in range(12):
#     plt.subplot(12,1,r+1)
#     plt.plot(signal[r])

def find_seizure_time(file_no):
    a=np.where(mat['annotat_new'][0][file_no-1][0]==1)[0]
    s_time=[]
    e_time=[]
    if len(a)!=0:
        s_time.append(a[0])  
        for r in range(1,a.shape[0]):
            if a[r-1]-a[r]!=-1:
                e_time.append(a[r-1]+1)
                s_time.append(a[r]) 
        e_time.append(a[-1])  
        
    return np.array(s_time),np.array(e_time)

s,e=find_seizure_time(7)
print("s time",s)
print("e time",e)   
x=np.arange(0,signal.shape[1],1)
seizure=np.zeros((signal.shape[1],))
for r in range(len(s)):
    seizure[s[r]*256:e[r]*256+1]=0.0005

plt.plot(x,seizure)
plt.plot(x,signal[0])
# for r in range(len(s)):
#     plt.plot(seizure_time[f'nsample{r}'],seizure_time[f'time{r}'],'r')
plt.show()