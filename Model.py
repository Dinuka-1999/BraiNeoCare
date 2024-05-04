import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np 
from scipy import signal 
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
from keras import regularizers

x_train = np.load('Datasets/Processed_data/traindata.npy',mmap_mode='r')
y_train = np.load('Datasets/Processed_data/trainlabels.npy',mmap_mode='r')
x_test  = np.load('Datasets/Processed_data/testdata.npy',mmap_mode='r')
y_test  = np.load('Datasets/Processed_data/testlabels.npy',mmap_mode='r')

mean=x_train.mean()
std=x_train.std()

x_train=(x_train-mean)/std
x_test=(x_test-mean)/std

x_train=np.expand_dims(x_train,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)

np.random.seed(42)
train_indices = np.arange(x_train.shape[0])
np.random.shuffle(train_indices)
x_train = x_train[train_indices]
y_train = y_train[train_indices]

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[1], 'GPU')

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

class AttentionLayer(layers.Layer):
    def __init__(self, output_dim):
        super(AttentionLayer, self).__init__()
        self.output_dim = output_dim
    
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',shape=(input_shape[-1], self.output_dim), initializer='random_normal',trainable=True)
        self.WK = self.add_weight(name='WK',shape=(input_shape[-1], self.output_dim), initializer='random_normal',trainable=True)
        self.WV = self.add_weight(name='WV',shape=(input_shape[-1], self.output_dim), initializer='random_normal',trainable=True)

    def call(self, input,adj):
        Q = tf.matmul(input, self.WQ)
        K = tf.matmul(input, self.WK)
        V = tf.matmul(input, self.WV)
        e = tf.matmul(Q, K, transpose_b=True)
        e = e / tf.math.sqrt(tf.cast(self.output_dim, tf.float32))
        zero_mat= -1e20*tf.ones_like(e)
        msked_e=tf.where(adj==1.0,e,zero_mat)
        alpha = tf.nn.softmax(msked_e, axis=-1)
        H = tf.matmul(alpha, V)
        return H

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
    y- layers.Conv2D(64,(1,7),activation='relu',padding='same')(x)
    x= layers.add([x,y])
    x= layers.AveragePooling2D((1,2))(x)
    x= layers.BatchNormalization()(x)
    x= layers.SpatialDropout2D(0.2)(x)

    x= layers.Conv2D(8,(1,5),activation='relu',padding='same')(x)
    y= layers.Conv2D(8,(1,7),activation='relu',padding='same')(x)
    x= layers.add([x,y])
    x= layers.AveragePooling2D((1,2))(x)
    x= layers.BatchNormalization()(x)
    x= layers.SpatialDropout2D(0.2)(x)

    x= layers.Conv2D(1,(1,5),activation='relu',padding='same')(x)
    y= layers.Conv2D(1,(1,7),activation='relu',padding='same')(x)
    x= layers.add([x,y])
    x= layers.AveragePooling2D((1,2))(x)
    x= layers.Reshape((12,24))(x)

    x= GATLayer(37)(x,adj)
    x= layers.Dropout(0.2)(x)
    x= GATLayer(32)(x,adj)
    x= layers.Dropout(0.2)(x)
    x= GATLayer(16)(x,adj)
    
    x= layers.GlobalAveragePooling1D()(x)
    x= layers.Dropout(0.2)(x)
    x= layers.Dense(32,activation='relu')(x)
    x= layers.Dropout(0.2)(x)
    x= layers.Dense(16,activation='relu')(x)
    x= layers.Dropout(0.2)(x)
    x= layers.Dense(1,activation='sigmoid')(x)

    model = keras.Model(inputs=Input, outputs=x)

    optimizer=keras.optimizers.Adam(learning_rate=0.002)
    loss=keras.losses.BinaryFocalCrossentropy(from_logits=False,gamma=2,alpha=2,apply_class_balancing=True)
    kappa=tfa.metrics.CohenKappa(num_classes=2)
    fp=keras.metrics.FalsePositives()
    tn=keras.metrics.TrueNegatives()
    precision = keras.metrics.Precision(name='precision')
    recall = keras.metrics.Recall(name='recall')
    AUROC = keras.metrics.AUC(curve='ROC', name = 'AUROC')
    AUPRC = keras.metrics.AUC(curve='PR', name = 'AUPRC')
    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy', AUROC, AUPRC,fp,tn, precision, recall,kappa])   
    return model
model=create_model()

checkpoint_path = "GAT_new/cp_{epoch:02d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path) 
cp_callback=keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=False,verbose=0,save_best_only=True,monitor='val_AUROC',mode='max')  
history=model.fit(x_train,y_train,epochs=100,batch_size=512,verbose=1,validation_data=(x_test,y_test),callbacks=[cp_callback])

metrics=['accuracy','val_accuracy','loss','val_loss','val_AUROC','val_AUPRC','val_recall_','val_cohen_kappa']

epochs = range(1, 51)
fig,ax=plt.subplots(4,2,figsize=(20,20))
for r in range(8):
    ax[r//2][r%2].plot(history[metrics[r]],color='r')
    ax[r//2][r%2].set_title(metrics[r])
    ax[r//2][r%2].set_xlabel('Epochs')
    ax[r//2][r%2].legend()
    ax[r//2][r%2].grid()
fig.tight_layout()
plt.show()


