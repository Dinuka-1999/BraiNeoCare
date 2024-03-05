import tensorflow as tf 
from tensorflow import keras    
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import Read_Data as rs
import tqdm
from matplotlib.animation import FuncAnimation 
import tensorflow_addons as tfa

EEG,labels=rs.read_a_file("../BraiNeoCare/Datasets/zenodo_eeg/eeg.edf",19,plot_=True)

model = keras.models.load_model("GAT_correct_7\cp_23.ckpt")
model.layers[-1].activation = None

model.summary()

grad_model = keras.Model(model.inputs,[model.get_layer("gat_layer_5").output,model.output])

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

mean,std=np.load("mean_std.npy")

fig,ax=plt.subplots(12,1,figsize=(20,20))
lb=9500
ub=9500+30*32
for r in range(12):
    ax[r].plot(EEG[r][lb:ub])
    ax[r].plot(labels[r][lb:ub])
    # ax[r].set_title(channel_names[r])
fig.tight_layout()
plt.show()

channel_names=["Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2","Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"]

lb=9984+384*2
ub=9984+384*2+384
x=PreprocesSignal(EEG[:,lb:ub],mean,std)
h,p=GradCAM(x)
print("prediction:- ",1 if p>=0.5 else 0)

resized_heatmap = cv.resize(h, (384,12), interpolation=cv.INTER_LINEAR)
fig,ax=plt.subplots(12,1,figsize=(20,25))
for r in range(12):
    ax[r].plot(EEG[r,lb:ub],color='k')
    im=ax[r].imshow(resized_heatmap[r].reshape(1,384),cmap='bwr',alpha=1,extent=[0,384,EEG[r,lb:ub].min(),EEG[r,lb:ub].max()],aspect='auto',vmax=1,vmin=0)
    ax[r].set_title(channel_names[r])   
fig.tight_layout()
cbar=fig.colorbar(im,ax=ax,orientation='horizontal',pad=0.02,shrink=0.5)
cbar.set_label('Relevance')
plt.show()