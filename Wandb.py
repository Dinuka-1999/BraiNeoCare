import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping,ModelCheckpoint
from scipy import signal
import os 
import pandas as pd
import wandb
from wandb.keras import WandbCallback
from numba import cuda

x_train=np.load('../BraiNeoCare/Datasets/balanced_x_train_common_4s.npy', mmap_mode='r')[0:10000,...]
x_test=np.load('../BraiNeoCare/Datasets/balanced_x_test_common_4s.npy', mmap_mode='r')
y_train=np.load('../BraiNeoCare/Datasets/balanced_y_train_common_4s.npy', mmap_mode='r')[0:10000,...]
y_test=np.load('../BraiNeoCare/Datasets/balanced_y_test_common_4s.npy', mmap_mode='r')

mean=x_train.mean()
std=x_train.std()
x_train=(x_train-mean)/std
x_test=(x_test-mean)/std

np.random.seed(42)
train_indices = np.arange(x_train.shape[0])
np.random.shuffle(train_indices)
x_train = x_train[train_indices].reshape(x_train.shape[0],1024,12)
y_train = y_train[train_indices]
test_indices = np.arange(x_test.shape[0])   
np.random.shuffle(test_indices)
x_test = x_test[test_indices].reshape(x_test.shape[0],1024,12)
y_test = y_test[test_indices]

sweep_config = {
    'method': 'grid',
    'metric': {
      'name': 'val_loss',
      'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [256,128,64,32]
        },
        'dropout': {
            'values': [0.1, 0.2, 0.25, 0.3, 0.5]
        },
        'spatdropout': {
            'values': [0.1, 0.2, 0.25, 0.3, 0.5]
        },
        'learning_rate': {
            'values': [ 0.001, 0.002, 0.0025, 0.005, 0.01, 0.1]
        },
        'weight_decay': {
            'values': [ 0.001, 0.002, 0.0025, 0.005, 0.01, 0.1]
        },
        'optimizer': {
            'values': ['adam', 'nadam','sgd','rmsprop']
        },
        'activation': {
            'values': ['relu', 'elu', 'swish']
        },
        'kernel_size': {
            "values": [[5],[7],[5,7]]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="BT_new_1")

def model_train():
    
    config_defaults = {
        'batch_size': 128,
        'dropout': 0.25,
        'spatdropout': 0.25,
        'learning_rate': 0.0025,
        'weight_decay': 0.01,
        'optimizer': 'adam',
        'activation': 'relu',
        'kernel_size': [5]
    }
    wandb.init(config=config_defaults)
    config = wandb.config

    Input=keras.Input(shape=(1024,12),name="Input_signal")
    x=layers.Conv1D(32,3,padding="same",activation=config.activation)(Input)
    y=layers.Conv1D(32,5,padding='same',activation=config.activation)(Input)
    a=layers.add([x,y])
    x=layers.MaxPooling1D(4)(a)
    x=layers.SpatialDropout1D(config.spatdropout)(x) 

    x=layers.Conv1D(64,3,padding="same",activation=config.activation)(x) 
    y=x
    for r in config.kernel_size:
        y=layers.Conv1D(64,r,padding='same',activation=config.activation)(y)

    a=layers.add([x,y])
    
    x=layers.GlobalAveragePooling1D()(a)
    x=layers.Dropout(config.dropout)(x)
    x=layers.Dense(32,activation='swish')(x) 
    x=layers.Dense(16,activation='swish')(x)
    x=layers.Dense(1,activation='sigmoid')(x)

    model1=keras.Model(inputs=Input,outputs=x)

    if config.optimizer=='adam':
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate,weight_decay=config.weight_decay)
    elif config.optimizer=='nadam':
        optimizer=keras.optimizers.Nadam(learning_rate=config.learning_rate,weight_decay=config.weight_decay)
    elif config.optimizer=='sgd':
        optimizer=keras.optimizers.SGD(learning_rate=config.learning_rate,weight_decay=config.weight_decay)
    elif config.optimizer=='rmsprop':
        optimizer=keras.optimizers.RMSprop(learning_rate=config.learning_rate,weight_decay=config.weight_decay)

    F1 = keras.metrics.FBetaScore()
    AUROC = keras.metrics.AUC(curve='ROC', name = 'AUROC')
    AUPRC = keras.metrics.AUC(curve='PR', name = 'AUPRC')
    model1.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy', F1, AUROC, AUPRC])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',min_delta=0.001)
    model1.fit(x_train,y_train,epochs=500,batch_size=config.batch_size,validation_data=(x_test,y_test),
               callbacks=[WandbCallback(save_model=False,verbosity=0,save_graph=False),early_stopping],verbose=0)
    
    tf.keras.backend.clear_session()  
    del model1
    cuda.select_device(0)
    cuda.close()
    os.system('clear')

wandb.agent(sweep_id, model_train)