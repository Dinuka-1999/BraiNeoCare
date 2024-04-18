import tensorflow as tf 
from tensorflow import keras    
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import Read_Data as rs
import tqdm
from matplotlib.animation import FuncAnimation 
import tensorflow_addons as tfa

#See the Read_Data.py file for the function read_a_file and more details
EEG,labels=rs.read_a_file("...Path to your EEG data file(.edf)...","...File number...",plot_=True)

model = keras.models.load_model("...Path to your model...")
model.layers[-1].activation = None

model.summary()

grad_model = keras.Model(model.inputs,[model.get_layer("...Name of the final GAT layer...").output,model.output])

def PreprocesSignal(signal,mean,std):
    signal=(signal-mean)/std
    signal=np.expand_dims(signal,axis=-1)
    signal=np.expand_dims(signal,axis=0)
    return signal

def GradCAM(signal,model=grad_model):
    
    with tf.GradientTape() as tape:
        CNN_outputs, predictions = grad_model(signal)
        Class=predictions[:,0]

    grads = tape.gradient(Class, CNN_outputs)

    heatmap = CNN_outputs[0] * tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = tf.nn.relu(heatmap)/tf.reduce_max(heatmap)
    # print(tf.reduce_mean(grads, axis=(0,1,2)))
    return heatmap.numpy(),tf.nn.sigmoid(predictions[0][0]).numpy()

mean,std=np.load("...Path to mean and std file...")#mean and std of the training data

fig,ax=plt.subplots(12,1,figsize=(20,20))

lb= 64600#Lower bound of the required time frame
ub= 64600+1152#Upper bound of the required time frame (1152 samples = 32 seconds)
for r in range(12):
    ax[r].plot(EEG[r][lb:ub])
    ax[r].plot(labels[r][lb:ub])
    # ax[r].set_title(channel_names[r])
fig.tight_layout()
plt.show()

channel_names=["Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2","Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"]

l=64600
u=64600+1152
h_map=np.zeros((12,1152))

for r in range(3):
    lb=l+384*r
    ub=l+384*r+384
    x=PreprocesSignal(EEG[:,lb:ub],mean,std)
    h,p=GradCAM(x)
    print("prediction:- ",1 if p>=0.5 else 0)
    resized_heatmap = cv.resize(h, (384,12), interpolation=cv.INTER_LINEAR)
    h_map[:,384*r:384*r+384]=resized_heatmap

fig,ax=plt.subplots(12,1,figsize=(20,25))
for r in range(12):
    ax[r].plot(EEG[r,l:u],color='k')
    im=ax[r].imshow(h_map[r].reshape(1,1152),cmap='bwr',alpha=0.8,extent=[0,1152,EEG[r,l:u].min(),EEG[r,l:u].max()],aspect='auto',vmax=1,vmin=0)
    ax[r].set_title(channel_names[r])   
fig.tight_layout()
cbar=fig.colorbar(im,ax=ax,orientation='horizontal',pad=0.02,shrink=0.5)
cbar.set_label('Relevance')
plt.show()