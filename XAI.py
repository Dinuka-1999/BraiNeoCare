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
cmap = plt.cm.bwr

l=8100
u=8100+1152
h_map=np.zeros((12,1152))
for r in range(3):
    lb=8100+384*r
    ub=8100+384*r+384
    x=PreprocesSignal(EEG[:,lb:ub],mean,std)
    h,p=GradCAM(x)
    print("prediction:- ",1 if p>=0.5 else 0)
    if p>=0.5:
        resized_heatmap = cv.resize(h, (384,12), interpolation=cv.INTER_LINEAR)
        h_map[:,384*r:384*r+384]=resized_heatmap
        
# plt.style.use('dark_background')
fig,ax=plt.subplots(12,1,figsize=(20,25))
norm = plt.Normalize(vmin=0, vmax=1)
b=np.arange(0,36,1/32)
for r in range(12):
    for i in range(1151):
        ax[r].plot([i/32,(i+1)/32],[EEG[r,l+i]*1e6,EEG[r,l+i+1]*1e6],color=cmap(norm(h_map[r,i])))
    ax[r].set_title(channel_names[r],fontsize=20)   
    ax[r].set_xlim([0,36])
    ax[r].tick_params(axis='x', labelsize=16)
    ax[r].tick_params(axis='y', labelsize=16)
ax[-1].set_xlabel("Time (s)",fontsize=20)
fig.tight_layout()
fig.text(-0.015, 0.6, 'Amplitude (uV)', va='center', rotation='vertical', fontsize=20)
im=ax[0].imshow(h_map[0].reshape(1,1152),cmap=cmap,alpha=1,extent=[0,36,EEG[0,l:u].min()*1e6,EEG[0,l:u].max()*1e6],aspect='auto',vmax=1,vmin=0,visible=False)
cbar=fig.colorbar(im,ax=ax,orientation='horizontal',pad=0.04,shrink=0.5)
cbar.set_label('Relevance',fontsize=20)
plt.show()