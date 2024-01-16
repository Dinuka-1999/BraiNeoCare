import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np 
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
        'loss':{
            'values':['focal','CE']
        },
        'cnn_block':{
            'values':[2,3,4]
        },
        'GAT_size':{
            'values':[[37,32,16],[48,32,16],[64,48,32]]
        }
    }
}

sweep_id=wandb.sweep(sweep_config,project='ablation_study_GAT')

x1=np.load('../BraiNeoCare/Datasets/GAT/zenodo_data_consensus_GAT.npy', mmap_mode='r')
y1=np.load('../BraiNeoCare/Datasets/GAT/zenodo_labels_consensus_GAT.npy', mmap_mode='r')

x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=42)
mean=x_train.mean()
std=x_train.std()
x_train=(x_train-mean)/std
x_test=(x_test-mean)/std


x_train=np.expand_dims(x_train,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)

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
    
channel_names=["Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2","Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"]
indices =[[r,i] for r,c1 in enumerate(channel_names) for i,c2 in enumerate(channel_names) if (c1.split("-")[0]==c2.split("-")[1] or c1.split("-")[1]==c2.split("-")[1] 
        or c1.split("-")[0]==c2.split("-")[0] or c1.split("-")[1]==c2.split("-")[0])]
adj=np.zeros((12,12))
for i in indices:
    adj[i[0]][i[1]]=1
adj=tf.constant(adj,dtype=tf.float32)


def model_train():

    wandb.init(config={
        'loss':'focal',
        'cnn_block':4,
        'GAT_size':[37,32,16]
    })
    config=wandb.config

    Input= keras.Input(shape=(12,384,1))

    if config.cnn_block==2:
        
        x= layers.Conv2D(32,(1,3),activation='relu',padding='same')(Input)
        y= layers.Conv2D(32,(1,5),activation='relu',padding='same')(Input)
        x= layers.add([x,y])
        x= layers.MaxPooling2D((1,2))(x)
        x= layers.BatchNormalization()(x)
        x= layers.SpatialDropout2D(0.1)(x)

        x= layers.Conv2D(1,(1,3),activation='relu',padding='same')(x)
        y= layers.Conv2D(1,(1,5),activation='relu',padding='same')(x)
        x= layers.add([x,y])
        x= layers.MaxPooling2D((1,2))(x)
        x= layers.Reshape((12,96))(x)
    
    elif config.cnn_block==3:
        x= layers.Conv2D(32,(1,3),activation='relu',padding='same')(Input)
        y= layers.Conv2D(32,(1,5),activation='relu',padding='same')(Input)
        x= layers.add([x,y])
        x= layers.MaxPooling2D((1,2))(x)
        x= layers.BatchNormalization()(x)
        x= layers.SpatialDropout2D(0.1)(x)

        x= layers.Conv2D(64,(1,3),activation='relu',padding='same')(x)
        y= layers.Conv2D(64,(1,5),activation='relu',padding='same')(x)
        x= layers.add([x,y])
        x= layers.MaxPooling2D((1,2))(x)
        x= layers.BatchNormalization()(x)
        x= layers.SpatialDropout2D(0.1)(x)

        x= layers.Conv2D(1,(1,3),activation='relu',padding='same')(x)
        y= layers.Conv2D(1,(1,5),activation='relu',padding='same')(x)
        x= layers.add([x,y])
        x= layers.MaxPooling2D((1,2))(x)
        x= layers.Reshape((12,48))(x)

    elif config.cnn_block==4:
        x= layers.Conv2D(32,(1,3),activation='relu',padding='same')(Input)
        y= layers.Conv2D(32,(1,5),activation='relu',padding='same')(Input)
        x= layers.add([x,y])
        x= layers.MaxPooling2D((1,2))(x)
        x= layers.BatchNormalization()(x)
        x= layers.SpatialDropout2D(0.1)(x)

        x= layers.Conv2D(64,(1,3),activation='relu',padding='same')(x)
        y= layers.Conv2D(64,(1,5),activation='relu',padding='same')(x)
        x= layers.add([x,y])
        x= layers.MaxPooling2D((1,2))(x)
        x= layers.BatchNormalization()(x)
        x= layers.SpatialDropout2D(0.1)(x)

        x= layers.Conv2D(8,(1,3),activation='relu',padding='same')(x)
        y= layers.Conv2D(8,(1,5),activation='relu',padding='same')(x)
        x= layers.add([x,y])
        x= layers.MaxPooling2D((1,2))(x)
        x= layers.BatchNormalization()(x)
        x= layers.SpatialDropout2D(0.1)(x)

        x= layers.Conv2D(1,(1,3),activation='relu',padding='same')(x)
        y= layers.Conv2D(1,(1,5),activation='relu',padding='same')(x)
        x= layers.add([x,y])
        x= layers.MaxPooling2D((1,2))(x)
        x= layers.Reshape((12,24))(x)

    val=config.GAT_size
    x= GATLayer(val[0])(x,adj)
    x= GATLayer(val[1])(x,adj)
    x= GATLayer(val[2])(x,adj)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x= layers.Dense(32,activation='relu')(x)
    x= layers.Dropout(0.1)(x)
    x = layers.Dense(16,activation='relu')(x)
    x = layers.Dense(1,activation='sigmoid')(x)

    model = keras.Model(inputs=Input, outputs=x)
    optimizer=keras.optimizers.Adam(learning_rate=0.002,weight_decay=0.0025)

    if config.loss=='CE':
        loss=keras.losses.BinaryCrossentropy(from_logits=False)
    else:
        loss=keras.losses.BinaryFocalCrossentropy(from_logits=False,gamma=2,alpha=0.4,apply_class_balancing=True)
    
    precall = keras.metrics.Precision()
    recall = keras.metrics.Recall()
    AUROC = keras.metrics.AUC(curve='ROC', name = 'AUROC')
    AUPRC = keras.metrics.AUC(curve='PR', name = 'AUPRC')

    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy',AUROC, AUPRC, precall, recall])    
  
    history=model.fit(x_train,y_train,epochs=200,batch_size=512,verbose=1,validation_data=(x_test,y_test),callbacks=[WandbCallback(save_model=True,verbosity=0,save_graph=True)])

wandb.agent(sweep_id,model_train)