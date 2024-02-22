import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np 
import scipy
import pandas as pd
import os
import tensorflow_addons as tfa
from keras import regularizers
import Read_Data as RD

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

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
        zero_mat= -1e9*tf.ones_like(e)
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
    x= layers.Dense(32,activation='relu',kernel_regularizer=regularizer_dense)(x)
    x= layers.Dropout(0.2)(x)
    x= layers.Dense(16,activation='relu',kernel_regularizer=regularizer_dense)(x)
    x= layers.Dropout(0.2)(x)
    x= layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizer_dense)(x)

    model = keras.Model(inputs=Input, outputs=x)

    optimizer=keras.optimizers.Adam(learning_rate=0.002)
    loss=keras.losses.BinaryFocalCrossentropy(from_logits=False,gamma=2,alpha=0.4,apply_class_balancing=True)
    kappa=tfa.metrics.CohenKappa(num_classes=2)
    fp=keras.metrics.FalsePositives()
    tn=keras.metrics.TrueNegatives()
    precision = keras.metrics.Precision()
    recall = keras.metrics.Recall()
    AUROC = keras.metrics.AUC(curve='ROC', name = 'AUROC')
    AUPRC = keras.metrics.AUC(curve='PR', name = 'AUPRC')
    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy', AUROC, AUPRC,fp,tn, precision, recall,kappa])   
    return model

folder='../BraiNeoCare/Datasets/zenodo_eeg/'
files=os.listdir(folder)

for r in range(10):
    x_train,y_train,x_test,y_test=RD.read_data(folder,files,r*4+1,(r+1)*4)
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
    
    model=create_model()
    checkpoint_path = f"Saved_models/GAT_CV_10_{r}"+"/cp_{epoch:03d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path) 
    cp_callback=keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=False,verbose=0,save_best_only=True,monitor='val_AUROC',mode='max')
    history=model.fit(x_train,y_train,epochs=50,batch_size=512,verbose=1,validation_data=(x_test,y_test),callbacks=[cp_callback]) 
    with open(f"History/history_cv_10_{r}.jason", 'w') as f:
        pd.DataFrame(history.history).to_json(f)
    