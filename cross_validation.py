import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np 
from scipy import signal 
import pandas as pd
import os
import tensorflow_addons as tfa
from keras import regularizers

channel_names=["Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2","Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"]
indices =[[r,i] for r,c1 in enumerate(channel_names) for i,c2 in enumerate(channel_names) if (c1.split("-")[0]==c2.split("-")[1] or c1.split("-")[1]==c2.split("-")[1] 
          or c1.split("-")[0]==c2.split("-")[0] or c1.split("-")[1]==c2.split("-")[0])]
adj=np.zeros((12,12))
for i in indices:
    adj[i[0]][i[1]]=1
adj=tf.constant(adj,dtype=tf.float32)

class GATLayer(layers.Layer):

    def __init__(self,output_dim):
        super(GATLayer, self).__init__()
        self.output_dim = output_dim
        self.Leakyrelu = layers.LeakyReLU(alpha=0.2)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='W',shape=(input_shape[-1], self.output_dim), initializer='random_normal',trainable=True)
        self.a = self.add_weight(name='a',shape=(2*self.output_dim, 1), initializer='random_normal',trainable=True)
    
    def call(self,input,adj):
        H= tf.matmul(input, self.W)
        h1=tf.tile(tf.expand_dims(H, axis=1), [1,12,1,1])
        h2=tf.tile(tf.expand_dims(H, axis=2), [1,1,12,1])
        result =tf.concat([h1 , h2], axis=-1)
        e=self.Leakyrelu(tf.squeeze(tf.matmul(result, self.a),axis=-1))
        zero_mat= -1e20*tf.ones_like(e)
        msked_e=tf.where(adj==1.0,e,zero_mat)
        alpha=tf.nn.softmax(msked_e,axis=-1)
        HPrime=tf.matmul(alpha,H)
        return tf.nn.elu(HPrime)

def create_model():
    Input= keras.Input(shape=(12,384,1))
    regularizer_dense=regularizers.l2(0.0001)

    x= layers.Conv2D(32,(1,5),activation='relu',padding='same')(Input)
    y= layers.Conv2D(32,(1,7),activation='relu',padding='same')(Input)
    x= layers.add([x,y])
    x= layers.AveragePooling2D((1,2))(x)
    x= layers.BatchNormalization()(x)
    x= layers.SpatialDropout2D(0.2)(x)

    x= layers.Conv2D(64,(1,5),activation='relu',padding='same')(x)
    y= layers.Conv2D(64,(1,7),activation='relu',padding='same')(x)
    x= layers.add([x,y])
    x= layers.AveragePooling2D((1,2))(x)
    x= layers.BatchNormalization()(x)
    x= layers.SpatialDropout2D(0.2)(x)

    x= layers.Conv2D(8,(1,5),activation='relu',padding='same')(x)
    y= layers.SpatialDropout2D(0.2)(x)
    y= layers.Conv2D(8,(1,7),activation='relu',padding='same')(y)
    x= layers.add([x,y])
    x= layers.AveragePooling2D((1,2))(x)
    x= layers.BatchNormalization()(x)
    x= layers.SpatialDropout2D(0.2)(x)

    x= layers.Conv2D(1,(1,5),activation='relu',padding='same')(x)
    y= layers.SpatialDropout2D(0.2)(x)
    y= layers.Conv2D(1,(1,7),activation='relu',padding='same')(y)
    x= layers.add([x,y])
    x= layers.AveragePooling2D((1,2))(x)
    x= layers.Reshape((12,24))(x)

    x= GATLayer(37)(x,adj)
    x= GATLayer(32)(x,adj)
    x= GATLayer(16)(x,adj)

    x= layers.GlobalAveragePooling1D()(x)
    x= layers.Dropout(0.2)(x)
    x= layers.Dense(32,activation='relu',kernel_regularizer=regularizer_dense)(x)
    x= layers.Dropout(0.2)(x)
    x= layers.Dense(16,activation='relu',kernel_regularizer=regularizer_dense)(x)
    x= layers.Dropout(0.2)(x)
    x= layers.Dense(1,activation='sigmoid')(x)

    model = keras.Model(inputs=Input, outputs=x)

    optimizer=keras.optimizers.Adam(learning_rate=0.002)
    loss=keras.losses.BinaryFocalCrossentropy(from_logits=False,gamma=2,alpha=0.4,apply_class_balancing=True)
    # loss=keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0.001)  
    kappa=tfa.metrics.CohenKappa(num_classes=2)
    fp=keras.metrics.FalsePositives()
    tn=keras.metrics.TrueNegatives()
    precision = keras.metrics.Precision()
    recall = keras.metrics.Recall()
    AUROC = keras.metrics.AUC(curve='ROC', name = 'AUROC')
    AUPRC = keras.metrics.AUC(curve='PR', name = 'AUPRC')
    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy', AUROC, AUPRC,fp,tn, precision, recall,kappa])   
    return model

import mne
import os
import scipy
import spkit as sp

mat=scipy.io.loadmat('../BraiNeoCare/Datasets/zenodo_eeg/annotations_2017.mat')

def signal_array(raw_data):
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

l_cut=16 #30
h_cut=0.5 #1

b_low,a_low=scipy.signal.butter(order,l_cut,'low',fs=fs)  #cheby2 20
b_high,a_high=scipy.signal.butter(order,h_cut,'high',fs=fs) #cheby2 20

def bandpassFilter(signal1):
    lowpasss=scipy.signal.filtfilt(b_low,a_low,signal1,axis=1)
    return scipy.signal.filtfilt(b_high,a_high,lowpasss,axis=1)

def downsample(signal1):
    filter_signal=bandpassFilter(signal1)
    new_n_samples=int(signal1.shape[1]*resampling_factor)
    return scipy.signal.resample(filter_signal,new_n_samples,axis=1)

def check_consecutive_zeros(arr):
    return np.all(arr==0)

folder='../BraiNeoCare/Datasets/zenodo_eeg/'
files=os.listdir(folder)

def read_data(low,high):
    train_signals=[]
    train_seizure=[]
    test_signals=[]
    test_seizure=[]
    c=1
    for file in files:
        if file.endswith('.edf'):
            
            s_Time,e_Time=find_seizure_time(int(file.split('.')[0][3:])) 

            if (len(s_Time)!=0 ):
                data = mne.io.read_raw_edf(folder+file,verbose=False)
                try:
                    data=data.pick_channels(['EEG Fp1-Ref', 'EEG Fp2-Ref',  'EEG C3-Ref', 'EEG C4-Ref', 'EEG Cz-Ref', 'EEG T3-Ref',
                                            'EEG T4-Ref', 'EEG O1-Ref', 'EEG O2-Ref'],ordered=True)
                    raw_data=data.get_data()
                    signal1=signal_array(raw_data)
                except:
                    data=data.pick_channels(['EEG Fp1-REF', 'EEG Fp2-REF', 'EEG C3-REF','EEG C4-REF', 'EEG Cz-REF', 'EEG T3-REF', 
                                            'EEG T4-REF', 'EEG O1-REF', 'EEG O2-REF'],ordered=True)
                    raw_data=data.get_data()
                    signal1=signal_array(raw_data)
                
                # signal=removeArtifacts(signal1)
                signal=downsample(signal1)
                u_bound=384
                l_bound=0
                started=False

                while u_bound<signal.shape[1]:
                    
                    partition=signal[:,l_bound:u_bound]

                    if np.any((s_Time*32>=l_bound) & (s_Time*32<=u_bound-1)):
                        if low<=c<high:
                            test_signals.append(partition)
                            test_seizure.append(1)
                        else:
                            train_signals.append(partition)
                            train_seizure.append(1)
                        
                        started=True
                        u_bound+=32
                        l_bound+=32

                    elif np.any((l_bound<(e_Time*32-1)) & ((e_Time*32-1)<=(u_bound-1))):
                        if low<=c<high:
                            test_signals.append(partition)
                            test_seizure.append(1)
                        else:
                            train_signals.append(partition)
                            train_seizure.append(1)
                        started=False
                        u_bound+=32
                        l_bound+=32

                    elif (started and np.any((e_Time*32-1)>u_bound-1)):
                        if low<=c<high:
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
                            if low<=c<high:
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

    return train_signals,train_seizure,test_signals,test_seizure

for r in range(5):
    x_train,y_train,x_test,y_test=read_data(r*8+1,(r+1)*8+1)
    mean=x_train.mean()
    std=x_train.std()
    x_train=(x_train-mean)/std
    x_test=(x_test-mean)/std

    x_train=np.expand_dims(x_train,axis=-1)
    x_test=np.expand_dims(x_test,axis=-1)

    np.random.seed(25)
    train_indices = np.arange(x_train.shape[0])
    np.random.shuffle(train_indices)
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    print(f"Start of {r}th fold")
    model=create_model()
    checkpoint_path = f"GAT_CV_{r}"+"/cp_{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path) 
    cp_callback=keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=False,verbose=0,save_best_only=True,monitor='val_accuracy')
    history=model.fit(x_train,y_train,epochs=50,batch_size=512,verbose=1,validation_data=(x_test,y_test),callbacks=[cp_callback]) 
    with open(f"history_cv_{r}.jason", 'w') as f:
        pd.DataFrame(history.history).to_json(f)
    print(f"Done for {r}th fold")