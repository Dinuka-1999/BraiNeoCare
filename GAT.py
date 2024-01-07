import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np 
from scipy import signal 
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback

sweep_config = {
    'method':'grid',
    'metric':{
        'name':'val_accuracy',
        'goal':'maximize'
    },
    'parameters':{
        'model':{
            'values':['1D','2D']
        },
        'leraning_rate':{
            'values':[0.001,0.002,0.0025,0.003,0.01,0.1]
        },
        'weight_decay':{
            'values':[0.0001,0.001,0.002,0.0025,0.005,0.01]
        }
    }
}

sweep_id=wandb.sweep(sweep_config,project='GAT')

x1=np.load('../BraiNeoCare/Datasets/GAT/zenodo_data_consensus_4s_GAT.npy', mmap_mode='r')
y1=np.load('../BraiNeoCare/Datasets/GAT/zenodo_labels_consensus_4s_GAT.npy', mmap_mode='r')

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
            self.LeakyReLU = layers.LeakyReLU(alpha=0.2)
        
        def build(self, input_shape):
            self.W = self.add_weight(name='W',shape=(input_shape[-1], self.output_dim), initializer='random_normal',trainable=True)
            self.a = self.add_weight(name='a',shape=(2*self.output_dim, 1), initializer='random_normal',trainable=True)
        
        def call(self,input,adj):
            H= tf.matmul(input, self.W)
            h1=tf.tile(tf.expand_dims(H, axis=1), [1,12,1,1])
            h2=tf.tile(tf.expand_dims(H, axis=2), [1,1,12,1])
            result =tf.concat([h1 , h2], axis=-1)
            e=self.LeakyReLU(tf.squeeze(tf.matmul(result, self.a),axis=-1))
            zero_mat=-1e20*tf.zeros_like(e)
            msked_e=tf.where(adj==1,e,zero_mat)
            alpha=tf.nn.softmax(msked_e,axis=-1)
            HPrime=tf.matmul(alpha,H)
            return tf.nn.elu(HPrime)

def model_train():

    wandb.init(config={
        'model':'1D',
        'learning_rate':0.001,
        'weight_decay':0.0001
    })
    config=wandb.config

    if config.model=='1D':
        x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=42)

        mean=x_train.mean()
        std=x_train.std()
        x_train=((x_train-mean)/std).reshape(x_train.shape[0],384,12)
        x_test=((x_test-mean)/std).reshape(x_test.shape[0],384,12)

        y_train=y_train.astype(np.float32)
        y_test=y_test.astype(np.float32)

        Input=keras.Input(shape=(384,12),name="Input_signal")
        x=layers.Conv1D(32,3,padding="same",activation='swish')(Input)
        y=layers.Conv1D(32,5,padding='same',activation='swish')(Input)
        a=layers.add([x,y])
        x=layers.MaxPooling1D(4)(a)
        x=layers.SpatialDropout1D(0.25)(x) 

        x=layers.Conv1D(12,3,padding="same",activation='swish')(x) 
        y=layers.Conv1D(12,5,padding='same',activation='swish')(x)#new
        a=layers.add([x,y])#new

        x= layers.Reshape((12,96))(a)

    elif config.model=='2D':
        x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=42)

        mean=x_train.mean()
        std=x_train.std()
        x_train=(x_train-mean)/std
        x_test=(x_test-mean)/std

        x_train=np.expand_dims(x_train,axis=-1)
        x_test=np.expand_dims(x_test,axis=-1)
        y_train=y_train.astype(np.float32)
        y_test=y_test.astype(np.float32)
         

        Input= keras.Input(shape=(12,384,1)) 

        x= layers.Conv2D(4,(1,5),activation='relu',padding='same')(Input)
        # x= layers.SpatialDropout2D(0.2)(x)
        x= layers.Conv2D(8,(1,5),activation='relu',padding='same')(x)
        # x= layers.concatenate([x,y])
        x= layers.MaxPool2D((1,2))(x)
        x= layers.BatchNormalization()(x)
        x= layers.SpatialDropout2D(0.2)(x)

        x= layers.Conv2D(16,(1,5),activation='relu',padding='same')(x)
        # x= layers.SpatialDropout2D(0.2)(x)
        x= layers.Conv2D(32,(1,5),activation='relu',padding='same')(x)
        # x= layers.concatenate([x,y])
        x= layers.MaxPool2D((1,2))(x)
        x= layers.BatchNormalization()(x)
        x= layers.SpatialDropout2D(0.2)(x)

        x= layers.Conv2D(8,(1,5),activation='relu',padding='same')(x)
        # x= layers.SpatialDropout2D(0.2)(x)
        x= layers.Conv2D(8,(1,5),activation='relu',padding='same')(x)
        # x= layers.add([x,y])
        x= layers.MaxPool2D((1,2))(x)
        x= layers.BatchNormalization()(x)
        x= layers.SpatialDropout2D(0.2)(x)

        x= layers.Conv2D(1,(1,5),activation='relu',padding='same')(x)
        # x= layers.SpatialDropout2D(0.2)(x)
        x= layers.Conv2D(1,(1,5),activation='relu',padding='same')(x)
        # x= layers.add([x,y])
        x= layers.MaxPool2D((1,2))(x)
        x= layers.Reshape((12,24))(x)

    x= GATLayer(37)(x,adj)
    x= GATLayer(32)(x,adj)
    x= GATLayer(16)(x,adj)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x= layers.Dense(32,activation='relu')(x)
    x= layers.Dropout(0.2)(x)
    x = layers.Dense(16,activation='relu')(x)
    x = layers.Dense(1,activation='sigmoid')(x)

    model = keras.Model(inputs=Input, outputs=x)

    optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate,weight_decay=config.weight_decay)
    loss=keras.losses.BinaryFocalCrossentropy(from_logits=False,gamma=2,alpha=0.4,apply_class_balancing=True)
    # F1=keras.metrics.F1Score(threshold=0.5, name='F1', dtype=None) 
    precall = keras.metrics.Precision()
    recall = keras.metrics.Recall()
    AUROC = keras.metrics.AUC(curve='ROC', name = 'AUROC')
    AUPRC = keras.metrics.AUC(curve='PR', name = 'AUPRC')
    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy',AUROC, AUPRC, precall, recall])    

    # checkpoint_path = "GAT_model/cp_{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path) 
    # cp_callback=keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=0,save_best_only=True,monitor='val_accuracy')  
    history=model.fit(x_train,y_train,epochs=100,batch_size=512,verbose=1,validation_data=(x_test,y_test),callbacks=[WandbCallback(save_model=True,verbosity=0,save_graph=False)])
wandb.agent(sweep_id,model_train)