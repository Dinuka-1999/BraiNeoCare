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
import Read_Data as rd

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


channel_names=["Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2","Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"]

EEG,labels=rd.read_a_file("file","n",pre_processing=False)
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
