import tensorflow as tf 
from tensorflow import keras    
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg
import sys
import threading
import mne
import scipy
import spkit as sp

## Channel Selection
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

## Annotation reading
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

## Signal filtering and artifact removal
fs=256
fs_d=32
order=7
resampling_factor=fs_d/fs

l_cut=16
h_cut=0.5

b_low,a_low=scipy.signal.butter(order,l_cut,'low',fs=fs)
b_high,a_high=scipy.signal.butter(order,h_cut,'high',fs=fs)

def bandpassFilter(signal1):
    lowpasss=scipy.signal.filtfilt(b_low,a_low,signal1,axis=1)
    return scipy.signal.filtfilt(b_high,a_high,lowpasss,axis=1)

def downsample(signal1):
    filter_signal=bandpassFilter(signal1)
    new_n_samples=int(signal1.shape[1]*resampling_factor)
    return scipy.signal.resample(filter_signal,new_n_samples,axis=1)

def removeArtifacts(signal1):
    beta_val=0.3
    signal1=bandpassFilter(signal1)
    signal1=scipy.signal.resample_poly(signal1,up=128,down=256,axis=1)
    signal1=signal1*3000000
    clean_array= sp.eeg.ATAR(signal1.T, wv='db15', winsize=128, beta=beta_val, thr_method='ipr', OptMode='soft',verbose=0)
    clean_array=clean_array.T/3000000
    signal1=scipy.signal.resample_poly(clean_array,up=32,down=128,axis=1)
    return signal1

channel_names=["Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2","Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"]

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

    s_Time,e_Time=find_seizure_time(n) 
    seizures=np.zeros((12,len(signal[0])))

    for i in range(12):
        for r in range(len(s_Time)):
            seizures[i,s_Time[r]*256:e_Time[r]*256]=signal[i,s_Time[r]*256:e_Time[r]*256]

    return signal,seizures

EEG,labels=read_file("./Datasets/zenodo_eeg/eeg9.edf",9)
updated_array=np.zeros((12,256*12))

app = QApplication([])
win = pg.GraphicsLayoutWidget(show=True)
# win.setBackground('w')
plot_c1=[]
plot_c2=[]

for i in range(6):
    plot_c1.append(win.addPlot(title=channel_names[i]))
    plot_c2.append(win.addPlot(title=channel_names[i+6]))
    plot_c1[i].setRange(xRange=[0,384],yRange=[-1e-4,1e-4])
    plot_c2[i].setRange(xRange=[0,384],yRange=[-1e-4,1e-4])
    win.nextRow()

graph_c1 = []
graph_c2 = []
seizure_c1=[]
seizure_c2=[]
heatmap_c1=[]
heatmap_c2=[]

for i in range(6):
    # heatmap_c1.append(pg.ImageItem())
    # heatmap_c2.append(pg.ImageItem())
    # plot_c1[i].addItem(heatmap_c1[i])
    # plot_c2[i].addItem(heatmap_c2[i])
    graph_c1.append(plot_c1[i].plot(pen='y',width=10))
    graph_c2.append(plot_c2[i].plot(pen='y',width=10))
    seizure_c1.append(plot_c1[i].plot(pen='r',width=10))
    seizure_c2.append(plot_c2[i].plot(pen='r',width=10))

def update_plot():
    for i in range(6):
        graph_c1[i].setData(y=EEG1[i])
        graph_c2[i].setData(y=EEG1[i+6])
        seizure_c1[i].setData(y=label_EEG1[i])
        seizure_c2[i].setData(y=label_EEG1[i+6])
        # heatmap_c1[i].setImage(hmap[i].reshape(1, 384))
        # heatmap_c2[i].setImage(hmap[i+6].reshape(1, 384))

        # # Set colormap and extent for images
        # # heatmap_c1[i].setLookupTable(pg.colormap.get('CET-L17'))
        # # heatmap_c2[i].setLookupTable(pg.colormap.get('CET-L17'))
        # heatmap_c1[i].setRect(pg.QtCore.QRectF(0,EEG1[i].max(), 384, -EEG1[i].ptp()))
        # heatmap_c2[i].setRect(pg.QtCore.QRectF(0,EEG1[i+6].max(), 384, -EEG[i+6].ptp()))


model = keras.models.load_model("GAT_model_97/cp_0192.ckpt")
model.layers[-1].activation = None

mean=np.load("mean.npy")
std=np.load("std.npy")

grad_model = keras.Model(model.inputs,[model.get_layer("gat_layer_2").output,model.output])

def PreprocesSignal(signal,mean,std):
    signal=(signal-mean)/std
    signal=np.expand_dims(signal,axis=-1)
    signal=np.expand_dims(signal,axis=0)
    return signal

def GradCAM(signal,model=grad_model):
    
    with tf.GradientTape() as tape:
        GAT_outputs, predictions = grad_model(signal)
        Class=predictions[:,0]

    grads = tape.gradient(Class, GAT_outputs)

    heatmap = GAT_outputs[0] *  tf.reduce_mean(grads, axis=(0,1,2))

    heatmap = tf.nn.relu(heatmap)/tf.reduce_max(heatmap)

    return heatmap.numpy(),tf.nn.sigmoid(predictions[0][0]).numpy()

l,u=0,256
c=0
labels_array=np.zeros((12,256*12))

def loop():
    global l,u,c,EEG1,label_EEG1,hmap
# fig,ax=plt.subplots(6,2,figsize=(20,20))
    while True:
        updated_array[:,:256*12-256]=updated_array[:,256:]
        updated_array[:,256*12-256:]=EEG[:,l:u]
        labels_array[:,:256*12-256]=labels_array[:,256:]
        labels_array[:,256*12-256:]=labels[:,l:u]
        EEG1=downsample(updated_array)
        label_EEG1=downsample(labels_array)
        # EEG2=removeArtifacts(updated_array)
        x1=PreprocesSignal(EEG1,mean,std)
        h,p=GradCAM(x1)
        prediction=1 if p>0.5 else 0    
        hmap = cv.resize(h, (384,12), interpolation=cv.INTER_LINEAR)
        # for r in range(12):
        #     ax[r%6][r//6].cla()
        #     ax[r%6][r//6].plot(EEG1[r])  
        #     ax[r%6][r//6].plot(label_EEG1[r]) 
        #     ax[r%6][r//6].set_ylim([-1e-4,1e-4])
        # t=np.arange(0,384)/32
        update_plot()
        c+=1
        l+=256
        u+=256
        if u>EEG.shape[1]:
            print("done")
            break
    # plt.pause(0.001)

timer = QtCore.QTimer()
timer.timeout.connect(update_plot)

if __name__ == '__main__':

    t1 = threading.Thread(target=loop)
    t1.start()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QApplication.instance().exec_()
