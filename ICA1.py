import numpy as np
from scipy.signal import periodogram as psd
import numpy as np
import pandas as pd
from scipy import interpolate
from tensorflow import keras
from scipy import signal
from mne.preprocessing import infomax
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns

def pol_to_cart(r,theta):
    """Converts polar coordinates to cartesian coordinates
    r: (float) - radius
    theta: (float) - angle in radians
    
    -----------------------------------------
    returns: (tuple of floats) - x,y
    
    -----------------------------------------"""
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    return x,y

def use_infomax(X):
    """Using the infomax algorithm to extract the independent components of the EEG data
    X: (2d numpy array) - the EEG data
    -----------------------------------------

    returns:
    S_: (2d numpy array) - the independent components
    A_: (2d numpy array) - the mixing matrix
    -----------------------------------------
    """
    print(X.shape)
    unmixing=infomax(X.T,extended=True,random_state=97,max_iter=500,verbose=False)
    mixing=np.linalg.pinv(unmixing.T)
    S_=np.dot(unmixing,X)
    return S_,unmixing,mixing

def make_topoplot(V1):
    """Creating a topoplot of the EEG data
    V1: (2d numpy array) - column data of the mixing matrix(inverse of the mixing matrix) of ICA decomposition
    -----------------------------------------
    
    returns: (2d numpy array) - the topoplot of the EEG data"""
    #normalize V1
    V1=(V1-np.min(V1))/(np.max(V1)-np.min(V1))

    Grid_scale=67
    Channels=["T4","FP2","C4","O2","O1","FP1","C3","T3"]
    #read the csv file 
    df=pd.read_csv("Montage.csv")
    #retreive the Number for channels corresponding to labels
    Labels=df["Labels"]
    Channel_number=[]
    Theta=[]
    Rad=[]
    for i in range(len(Channels)):
        ind=Labels.index[Labels==Channels[i]].to_list()[0]
        Channel_number.append(ind)
        Theta.append(df["Theta"][ind])
        Rad.append(df["Radius"][ind])
    #converting theta from degrees to radians
    Theta=[i*np.pi/180 for i in Theta]
    #converting the array to numpy array
    Theta=np.array(Theta)
    Rad=np.array(Rad)
    Channels=np.array(Channels)
    Channel_number=np.array(Channel_number)
    #transforming the data to cartesian coordinates
    X,Y=[],[]
    for i in range(len(Channels)):
        x,y=pol_to_cart(Rad[i],Theta[i])
        X.append(x)
        Y.append(y)
    X=np.array(X)
    Y=np.array(Y)
    xmin = -0.5
    xmax = 0.5
    ymin = -0.5
    ymax = 0.5
    xi,yi=np.meshgrid(np.linspace(xmin,xmax,Grid_scale),np.linspace(ymin,ymax,Grid_scale))
    #performing bilinear interpolation using RBFInterpolator
    rbf = interpolate.Rbf(X,Y,V1,function="inverse_multiquadric")
    grid_z0=rbf(xi,yi)
    #adding a mask of 0.5 radius
    mask=np.zeros((Grid_scale,Grid_scale))
    for i in range(Grid_scale):
        for j in range(Grid_scale):
            if np.sqrt((xi[i][j])**2+(yi[i][j])**2)<=0.5:
                mask[i][j]=1
            else:
                mask[i][j]=np.NaN
    grid_z0=grid_z0*mask
    mean_val=np.nanmean(grid_z0)
    #replace the NaN values with the mean
    grid_z0[np.isnan(grid_z0)]=mean_val
    return grid_z0

def get_spectral_power(X):
    """Calculating the spectral power of the EEG data
    X: (2d numpy array) - the EEG data
    -----------------------------------------
    
    returns: (2d numpy array) - the spectral power of the EEG data
    """
    f,X_power=signal.periodogram(X,fs=256,axis=1,nfft=128)
    return X_power

def evaluate_model(ICA,grids,psd,model):
    """ICA labeling model
    model input:
    1. Topoplot of the ICA components
    2. Spectral power of the ICA components
    3. ICA components
    -----------------------------------------
    returns: (1d numpy array) - the labels of the ICA components
    1 - brain signal
    0 - artefactual signal
    """
    #evaluate the model
    y_pred=model.predict([grids,psd,ICA],verbose=False)
    return y_pred

def label_ica(eeg_input,channel_names="chn8"):
    if channel_names=="chn8":
        #channel_names=["Fp1","Fp2","C3","C4","T3","T4","O1","O2"]
        channel_names=["T4","Fp2","C4","O2","O1","Fp1","C3","T3"]
        #'T4-Cz','Fp2-Cz','C4-Cz','O2-Cz','O1-Cz','Fp1-Cz','C3-Cz','T3-Cz'
    elif channel_names=="chn12":
        channel_names=["Fp1","Fp2","C3","C4","T3","T4","O1","O2","Fp1","Fp2","O1","O2"]
    else:
        raise ValueError("Invalid channel names")

    S_,unmixing,mixing=use_infomax(eeg_input)
    
    #testing for original signal

    S_power=get_spectral_power(S_)
    inv_A=unmixing.copy().T
    grids=[]
    for i in range(8):
        grids.append(make_topoplot(inv_A[i]))

    grids=np.array(grids)

    model=keras.models.load_model("IC_model_ICA_v15.h5")

    labels=np.array(evaluate_model(S_,grids,S_power,model))
    y_pred_app_2=np.zeros(labels.shape)
    y_pred_app_2[labels<=0.5]=1
    y_pred_app_2[labels>0.5]=0
    labels=y_pred_app_2

    eeg_clean=eeg_input.copy()
    eeg_artefact=eeg_input.copy()

    S_clean=S_.copy()
    S_artefact=S_.copy()

    for i in range(8):
        if labels[i]==0:
            S_artefact[i]=S_artefact[i]*0
        else:
            S_clean[i]=S_clean[i]*0

    eeg_clean=S_clean.T.dot(mixing).T
    eeg_artefact=S_artefact.T.dot(mixing).T
    #print(labels)
    return eeg_clean,eeg_artefact

def crop_signal(eeg_input,channel_names="chn8"):
    crop_interval=256
    las_sample=crop_interval*(eeg_input.shape[1]//crop_interval)
    eeg_input=eeg_input[:,:las_sample]
    return eeg_input

def get_artefacts(eeg_input,channel_names="chn8"):
    eeg_input=signal.resample_poly(eeg_input,up=256,down=500,axis=1)
    sos=signal.butter(4,[1,100],"bandpass",fs=256,output="sos")
    eeg_input=signal.sosfilt(sos,eeg_input)
    eeg_input=crop_signal(eeg_input,channel_names)
    #eeg_input=twelve_to_eight(eeg_input,channel_names)
    #break into 2 minute epochs
    clean_array=np.zeros(eeg_input.shape)
    artefact_array=np.zeros(eeg_input.shape)
    overlap_array=np.zeros((1,eeg_input.shape[1]))

    #determining time gap
    if eeg_input.shape[1]>2*60*256:
        T=eeg_input.shape[1]
        t=2*60*256
        a = np.ceil(T/t)
        overlap = int((a*t-T)/(a-1))
        overlap_gap = t-overlap
        """print("T : ",T//256)
        print("t : ",t//256)
        print("a : ",a)
        print("overlap : ",overlap//256)
        print("overlap_gap : ",overlap_gap//256)"""
    elif eeg_input.shape[1]<2*60*256:
        raise ValueError("Input data is less than 2 minutes")
    else:
        overlap=0
        t=2*60*256
        overlap_gap=t-overlap
    #plot eeg data, clean data and artefactual data
    if overlap==0:
        end_point=eeg_input.shape[1]
    else:
        end_point=eeg_input.shape[1]-overlap_gap
    for i in range(0,end_point,overlap_gap):      #120*256
        epoch=eeg_input[:,i:i+120*256]
        overlap_array[:,i:i+120*256]=overlap_array[:,i:i+120*256]+1
        eeg_clean,eeg_artefact=label_ica(epoch,channel_names)
        clean_array[:,i:i+120*256]=clean_array[:,i:i+120*256]+eeg_clean
        artefact_array[:,i:i+120*256]=artefact_array[:,i:i+120*256]+eeg_artefact
    overlap_array=1/overlap_array
    clean_array=np.multiply(clean_array,overlap_array)
    artefact_array=np.multiply(artefact_array,overlap_array)

    #resample to 250 Hz
    eeg_input=signal.resample_poly(eeg_input,up=250,down=256,axis=1)
    clean_array=signal.resample_poly(clean_array,up=250,down=256,axis=1)
    artefact_array=signal.resample_poly(artefact_array,up=250,down=256,axis=1)

    return eeg_input,clean_array,artefact_array

if __name__=="__main__":
    data=np.load("bc1.npy")
    eeg_input=data[0]
    eeg_input,clean_array,artefact_array=get_artefacts(eeg_input)
    print(eeg_input.shape,clean_array.shape,artefact_array.shape)
    plt.figure(figsize=(10,10))
    plt.plot(eeg_input[0])
    plt.plot(clean_array[0])
    plt.plot(artefact_array[0])
    plt.show()
   
