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


channel_names=["Fp1 - T3","T3 - O1","Fp1 - C3","C3 - O1","Fp2 - C4","C4 - O2","Fp2 - T4","T4 - O2","T3 - C3","C3 - Cz","Cz - C4","C4 - T4"]
cmap = plt.cm.bwr

l=0
u=10*60*32-4800+192
True_labels=np.zeros((1,u-l))
True_labels[0,266*32-l:971*32-l]=1
Pred_labels=np.zeros((1,u-l))
h_map=np.zeros((12,u-l))
previous=0
for r in range(u//384):
    lb=l+r*384
    ub=l+r*384+384
    x=PreprocesSignal(EEG[:,lb:ub],mean,std)
    h,p=GradCAM(x)
    Pred_labels[0,r*384:(r+1)*384]=p
    if p>=0.61:
        resized_heatmap = cv.resize(h, (384,12), interpolation=cv.INTER_LINEAR)
        if previous==0:
            h_map[:,384*r:384*r+384]=resized_heatmap
            previous=1
        else:
            h_map[:,384*r:384*r+384]=resized_heatmap
            for i in range(12):
                if (h_map[i,384*r]<0.15 and h_map[i,384*r-1]>0.85):
                    for j in range(15):
                        h_map[i,384*r-j]=0.5+j*0.023
                        h_map[i,384*r+1+j]=0.5-j*0.023
            previous=1
    else:
        previous=0


# plt.style.use('yellow_background')
fig,ax=plt.subplots(14,1,figsize=(20,40))

time= np.arange(0,7.6,1/(32*60))
ax[0].plot(time,True_labels[0],label="True labels",color='purple')
ax[0].plot(time,Pred_labels[0],label="Predicted labels",color='green')
ax[0].set_xlim([0,7.5])
ax[0].set_title("Seizure Probability",fontsize=27)
ax[0].hlines(0.61,0,7.5,linestyles='dashed',color='black',label="Threshold")
ax[0].tick_params(axis='x', labelsize=20)
ax[0].tick_params(axis='y', labelsize=20)
ax[0].legend(fontsize=15)

norm = plt.Normalize(vmin=0, vmax=1)
b=np.arange(0,7.5,1/(32*60))
for r in range(1,13):
    for i in range(u-l):
        ax[r].plot([i/(32*60),(i+1)/(32*60)],[EEG[r-1,l+i]*1e6,EEG[r-1,l+i+1]*1e6],color=cmap(norm(h_map[r-1,i])))
    ax[r].set_title(channel_names[r-1],fontsize=27)   
    ax[r].set_xlim([0,7.5])
    # ax[r].set_ylim(-28,28)
    ax[r].tick_params(axis='x', labelsize=20)
    ax[r].tick_params(axis='y', labelsize=20)
ax[-1].set_xlabel("Time (min)",fontsize=30)
fig.text(-0.015, 0.6, 'Amplitude (uV)', va='center', rotation='vertical', fontsize=30)

for i in range(1152+384):
        ax[13].plot([i/(32*60),(i+1)/(32*60)],[EEG[0,8116-384+i]*1e6,EEG[0,8116-384+i+1]*1e6],color=cmap(norm(h_map[0,3316-384+4800+i])))
ax[13].set_xticks(np.linspace(0,48/60,4),np.linspace(4.02, 4.83, num=4)) 
ax[13].tick_params(axis='x', labelsize=20)
ax[13].tick_params(axis='y', labelsize=20)
ax[13].set_title("Zoomed in time window of Fp1 - T3",fontsize=27)
fig.tight_layout()

plt.subplots_adjust(hspace=0.5)
im=ax[1].imshow(h_map[0].reshape(1,u-l),cmap=cmap,alpha=1,extent=[0,7.5,EEG[0,l:u].min()*1e6,EEG[0,l:u].max()*1e6],aspect='auto',vmax=1,vmin=0,visible=False)
cbar=fig.colorbar(im,ax=ax,orientation='horizontal',pad=0.04,shrink=0.5)
cbar.ax.tick_params(labelsize=22)
cbar.set_label('Relevance',fontsize=30)
plt.savefig("EX_AI.png",bbox_inches='tight',dpi=200)
plt.show()