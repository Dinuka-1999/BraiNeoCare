import sys
import socket
import struct
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg
import threading
import time
from scipy import signal
import cv2
from scipy.fft import fft, fftfreq

UDP_IP = "192.168.177.143" # The IP that is printed in the serial monitor from the ESP32
SHARED_UDP_PORT = 4210
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
sock.connect((UDP_IP, SHARED_UDP_PORT))

# Create the PyQtGraph application
app = QApplication([])
win = pg.GraphicsLayoutWidget(show=True)
show = [3,4]
plot=[]
orig_plot = []
baseLine_plot=[]
alpha_plot=[]
fft_plot=[]

for j,i in enumerate(show):
    plot.append(win.addPlot(title=f'Channel {i}'))
    orig_plot.append(win.addPlot(title=f'Original {i}'))
    alpha_plot.append(win.addPlot(title=f'Decision {i}'))
    fft_plot.append(win.addPlot(title=f'FFT {i}'))
    # baseLine_plot.append(win.addPlot(title=f'base_Line {i}'))
    win.nextRow()

filtered_curve = []
orig_curve = []
alpha_curve = []
fft_curve = []
color = ['b','g','r','c','b','g','r','c','b']
for i in range(len(show)):
    filtered_curve.append(plot[i].plot(pen='g'))
    orig_curve.append(orig_plot[i].plot(pen='b'))
    alpha_curve.append(alpha_plot[i].plot(pen='r'))
    fft_curve.append(fft_plot[i].plot(pen='c'))
xdata = np.arange(1000)  # Number of data points to display on the plot
ydata = np.zeros((len(show),1000))
num_samples = 0
rate = 250

def nothing(x):
    pass

cv2.namedWindow('Thresholds')
cv2.resizeWindow('Thresholds', 600,250)
cv2.createTrackbar('Vertical_upper','Thresholds',1500 ,5000, nothing)
cv2.createTrackbar('Vertical_lower','Thresholds',2000 ,5000, nothing)
cv2.createTrackbar('Horizontal_upper','Thresholds',2500 ,5000, nothing)
cv2.createTrackbar('Horizontal_lower','Thresholds',2500 ,5000, nothing)

def sliders():
    global v_up,v_low,h_up,h_low
    while True:
        v_up = cv2.getTrackbarPos('Vertical_upper','Thresholds')
        v_low = -cv2.getTrackbarPos('Vertical_lower','Thresholds')
        h_up = cv2.getTrackbarPos('Horizontal_upper','Thresholds')
        h_low = -cv2.getTrackbarPos('Horizontal_lower','Thresholds')
        time.sleep(0.001)

def loop():
    global ydata,num_samples
    while True:
        # Receive UDP packet
        data = sock.recv(4*9)
        value = struct.unpack('iiiiiiiii', data)

        for i,j in enumerate(show):
            ydata[i][:-1] = ydata[i][1:]
            ydata[i][-1] = value[j]

        num_samples += 1

def update_plot():
    for i in range(len(show)):
        orig_curve[i].setData(y=ydata_avg[i][100:])
        filtered_curve[i].setData(y=filtered[i][100:])
        alpha_curve[i].setData(y=alpha[i][100:])
        fft_curve[i].setData(y=yfft[i][0:1000//2], x=xfft)
        # decision_curve[i].setData(y=decision_arr[i])
        # upper_thresh_curve[i].setData(y=upper_thresh[i])
        # lower_thresh_curve[i].setData(y=lower_thresh[i])

def filter():
    global rate, filtered, baseLine_filtered,decision_arr,ydata_avg,upper_thresh,lower_thresh, alpha, yfft, xfft
    T = 1/rate
    xfft = fftfreq(1000, T)[:1000//2]
    while True:
        # if rate>80*2:
        lowpass = signal.butter(2, 80, 'lp', fs=rate, output='sos')
        bandstop = signal.butter(2, [45,55], 'bandstop', fs=rate, output='sos')
        # bandstop = signal.cheby1(2, 1, [45,55], 'bandpass', fs=rate, output='sos', analog=True)
        bandstop2 = signal.butter(2, [98,102], 'bandstop', fs=rate, output='sos')
        bandpass = signal.butter(2, [8,13], 'bandpass', fs=rate, output='sos')
        
        ydata_avg = ydata - np.mean(ydata,axis=1).reshape(-1,1)
        
        # filtered1=np.zeros((len(show),ydata.shape[1]))
        # for r,i in enumerate(show):
        filtered1 = signal.sosfilt(bandstop, ydata_avg)
        filtered2 = signal.sosfilt(bandstop2, filtered1)


        # filtered2 = np.zeros((len(show),ydata.shape[1]))
        # for r,i in enumerate(show):
        filtered3 = signal.sosfilt(lowpass, filtered2)
        filtered=np.zeros((len(show),ydata.shape[1]))
        for i in range(len(show)):
            filtered[i]=signal.medfilt(filtered3[i], kernel_size=9)
        alpha = signal.sosfilt(bandpass, filtered)

        yfft = np.real(fft(filtered))
            
            

# Set up a timer to update the plot
timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(10*len(show))  # Update every 10 ms

if __name__ == "__main__":
    sock.send('Hello ESP32'.encode())
    t1 = threading.Thread(target=loop)
    t1.start()
    t3 = threading.Thread(target=filter)
    t3.start()
    t5 = threading.Thread(target=sliders)
    t5.start()
    t6 = threading.Thread(target=update_plot)
    t6.start()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QApplication.instance().exec_()