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
from scipy.fft import fft, fftfreq
from tkinter import *


def update_menu():
    global lp_range_var, lp_order_var, hp_range_var, hp_order_var, bp_range_low_var, bp_range_high_var, bp_order_var, bs_range_low_var, bs_range_high_var, bs_order_var
    while True:
        lp_range_var = float(lp_range.get())
        lp_order_var = int(lp_order.get())
        hp_range_var = float(hp_range.get())
        hp_order_var = int(hp_order.get())
        bp_range_low_var = float(bp_range_low.get())
        bp_range_high_var = float(bp_range_high.get())
        bp_order_var = int(bp_order.get())
        bs_range_low_var = float(bs_range_low.get())
        bs_range_high_var = float(bs_range_high.get())
        bs_order_var = int(bs_order.get())

        time.sleep(0.001)

def menu():
    global lp_check_var, lp_range, lp_order, hp_check_var, hp_range, hp_order, bp_check_var, bp_range_low, bp_range_high, bp_order, bs_check_var, bs_range_low, bs_range_high, bs_order
    master = Tk()
    master.title("Menu")
    lp_frame = Frame(master)
    lp_frame.pack(side=TOP)
    hp_frame = Frame(master)
    hp_frame.pack(side=TOP)
    bp_frame = Frame(master)
    bp_frame.pack(side=TOP)
    bs_frame = Frame(master)
    bs_frame.pack(side=TOP)

    lp_check_var = IntVar()
    Checkbutton(lp_frame, text="Lowpass", variable=lp_check_var).pack(side=LEFT)
    Label(lp_frame, text="Range").pack(side=LEFT)
    lp_range = Entry(lp_frame)
    lp_range.insert(0, "80")
    lp_range.pack(side=LEFT)
    Label(lp_frame, text="Order").pack(side=LEFT)
    lp_order = Entry(lp_frame)
    lp_order.insert(0, "2")
    lp_order.pack(side=LEFT)

    hp_check_var = IntVar()
    Checkbutton(hp_frame, text="Highpass", variable=hp_check_var).pack(side=LEFT)
    Label(hp_frame, text="Range").pack(side=LEFT)
    hp_range = Entry(hp_frame)
    hp_range.insert(0, "0.5")
    hp_range.pack(side=LEFT)
    Label(hp_frame, text="Order").pack(side=LEFT)
    hp_order = Entry(hp_frame)
    hp_order.insert(0, "2")
    hp_order.pack(side=LEFT)

    bp_check_var = IntVar()
    Checkbutton(bp_frame, text="Bandpass", variable=bp_check_var).pack(side=LEFT)
    Label(bp_frame, text="Range").pack(side=LEFT)
    bp_range_low = Entry(bp_frame)
    bp_range_low.insert(0, "8")
    bp_range_low.pack(side=LEFT)
    bp_range_high = Entry(bp_frame)
    bp_range_high.insert(0, "13")
    bp_range_high.pack(side=LEFT)
    Label(bp_frame, text="Order").pack(side=LEFT)
    bp_order = Entry(bp_frame)
    bp_order.insert(0, "2")
    bp_order.pack(side=LEFT)

    bs_check_var = IntVar()
    Checkbutton(bs_frame, text="Bandstop", variable=bs_check_var).pack(side=LEFT)
    Label(bs_frame, text="Range").pack(side=LEFT)
    bs_range_low = Entry(bs_frame)
    bs_range_low.insert(0, "45")
    bs_range_low.pack(side=LEFT)
    bs_range_high = Entry(bs_frame)
    bs_range_high.insert(0, "55")
    bs_range_high.pack(side=LEFT)
    Label(bs_frame, text="Order").pack(side=LEFT)
    bs_order = Entry(bs_frame)
    bs_order.insert(0, "2")
    bs_order.pack(side=LEFT)

    mainloop()

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


def filter():
    global rate, filtered, ydata_avg, alpha, yfft, xfft
    T = 1/rate
    xfft = fftfreq(1000, T)[:1000//2]
    while True:
        # if rate>80*2:
        lowpass = signal.butter(lp_order_var, lp_range_var, 'lp', fs=rate, output='sos')
        highpass = signal.butter(hp_order_var, hp_range_var, 'hp', fs=rate, output='sos')
        bandstop = signal.butter(2, [45,55], 'bandstop', fs=rate, output='sos')
        bandpass = signal.butter(2, [8,13], 'bandpass', fs=rate, output='sos')
        
        ydata_avg = ydata - np.mean(ydata,axis=1).reshape(-1,1)
        filtered = ydata_avg
        
        if lp_check_var.get() == 1:
            filtered = signal.sosfilt(lowpass, ydata_avg)
        if hp_check_var.get() == 1:
            filtered = signal.sosfilt(highpass, filtered)
        if bp_check_var.get() == 1:
            filtered = signal.sosfilt(bandpass, filtered)
        if bs_check_var.get() == 1:
            filtered = signal.sosfilt(bandstop, filtered)
        
        # filtered=np.zeros((len(show),ydata.shape[1]))
        # for i in range(len(show)):
        #     filtered[i]=signal.medfilt(filtered3[i], kernel_size=9)
        alpha = signal.sosfilt(bandpass, filtered)

        yfft = np.abs(fft(filtered))
            
            

# Set up a timer to update the plot
timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(10*len(show))  # Update every 10 ms

if __name__ == "__main__":
    sock.send('Hello ESP32'.encode())
    t7 = threading.Thread(target=menu)
    t7.start()
    time.sleep(0.01)
    t8 = threading.Thread(target=update_menu)
    t8.start()
    time.sleep(0.01)
    t1 = threading.Thread(target=loop)
    t1.start()
    time.sleep(0.01)
    t3 = threading.Thread(target=filter)
    t3.start()
    time.sleep(0.01)
    t6 = threading.Thread(target=update_plot)
    t6.start()
    time.sleep(0.01)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QApplication.instance().exec_()