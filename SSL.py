import tensorflow as tf 
from tensorflow import keras
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
import cv2 as cv
import tqdm

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[1], 'GPU')

x_train=np.load('Datasets/CHB/chb_data.npy')
x_train=(x_train-x_train.mean())/x_train.std()
print(x_train.shape)
def transform1(signals):
    signals = np.flip(signals, axis=2)
    signals = np.roll(signals, 100, axis=2)
    for r in range(signals.shape[0]):
        random_noise=np.random.normal(0,0.001,(12,384))
        signals[r]+=random_noise
        n=np.random.randint(200,384,1)[0]
        start=np.random.randint(0,384-n)
        s=signals[r,:,start:start+200]
        signals[r]=cv.resize(s,(384,12),interpolation=cv.INTER_LINEAR)
    return signals

def transform2(signal,pieces=48):
    signal = signal*(-1)
    piece_n = 384//pieces
    for i in range(pieces):
        signal[:,:,i*piece_n:(i+1)*piece_n] = np.roll(signal[:,:,i*piece_n:(i+1)*piece_n], i%5, axis=2)
    for r in range(signal.shape[0]):
        random_noise=np.random.normal(0,0.005,(12,384))
        signal[r]+=random_noise
        n=np.random.randint(200,384,1)[0]
        start=np.random.randint(0,384-n)
        s=signal[r,:,start:start+200]
        signal[r]=cv.resize(s,(384,12),interpolation=cv.INTER_LINEAR)
    return signal

X_train_T1=transform1(x_train)
X_train_T2=transform2(x_train)

X_train_T1=np.expand_dims(X_train_T1,axis=-1)
X_train_T2=np.expand_dims(X_train_T2,axis=-1)

class contrastive_loss(keras.losses.Loss):
    def __init__(self,temperature=1.0):
        super(contrastive_loss,self).__init__()
        self.temperature=temperature

    def __call__(self,hidden1,hidden2,hidden_norm=True):
        LARGE_NUM = 1e9
        if hidden_norm:
            hidden1 = tf.math.l2_normalize(hidden1, -1)
            hidden2 = tf.math.l2_normalize(hidden2, -1)
        batch_size = tf.shape(hidden1)[0]
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

        logits_aa = tf.matmul(hidden1,hidden1, transpose_b=True) / self.temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / self.temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / self.temperature
        logits_ba = tf.matmul(hidden2,hidden1, transpose_b=True) / self.temperature

        loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ba, logits_bb], 1))
        loss = tf.reduce_mean(loss_a + loss_b)
        return loss
    
def CNN_encoder(Input):
    x= layers.Conv2D(32,(1,5),activation='relu',padding='same')(Input)
    y= layers.Conv2D(32,(1,7),activation='relu',padding='same')(Input)
    x= layers.add([x,y])
    x= layers.AveragePooling2D((1,2))(x)
    x= layers.BatchNormalization()(x)

    x= layers.Conv2D(64,(1,5),activation='relu',padding='same')(x)
    y= layers.Conv2D(64,(1,7),activation='relu',padding='same')(x)
    x= layers.add([x,y])
    x= layers.AveragePooling2D((1,2))(x)
    x= layers.BatchNormalization()(x)

    x= layers.Conv2D(8,(1,5),activation='relu',padding='same')(x)
    y= layers.Conv2D(8,(1,7),activation='relu',padding='same')(x)
    x= layers.add([x,y])
    x= layers.AveragePooling2D((1,2))(x)
    x= layers.BatchNormalization()(x)

    x= layers.Conv2D(1,(1,5),activation='relu',padding='same')(x)
    y= layers.Conv2D(1,(1,7),activation='relu',padding='same')(x)
    x= layers.add([x,y])
    x= layers.Flatten()(x)
    return x

def projection_head(encoder):
    x=layers.Dense(128,activation='relu')(encoder)
    x=layers.Dense(64)(x)
    return x

Input1=keras.Input(shape=(12,384,1))
Input2=keras.Input(shape=(12,384,1))
encoder1=CNN_encoder(Input1)
encoder2=CNN_encoder(Input2)
projection1=projection_head(encoder1)
projection2=projection_head(encoder2)

model=keras.Model(inputs=[Input1,Input2],outputs=[projection1,projection2]) 

optimizer=keras.optimizers.Adam(learning_rate=0.0005)
for epoch in range (200):
    for i in tqdm.tqdm(range(0,X_train_T1.shape[0],512),desc=f"epoch {epoch+1}/200"):
        with tf.GradientTape() as tape:
            hidden1,hidden2=model([X_train_T1[i:i+512],X_train_T2[i:i+512]])
            loss=contrastive_loss(1)(hidden1,hidden2)
        grads=tape.gradient(loss,model.trainable_weights)
        optimizer.apply_gradients(zip(grads,model.trainable_weights))
    print('loss:',loss.numpy())
model.save('Saved_models/SSL/SSL_model.keras')