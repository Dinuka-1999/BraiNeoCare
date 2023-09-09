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
    global lp_range_var, lp_order_var, hp_range_var, hp_order_var, bp_range_low_var, bp_range_high_var, bp_order_var, bs_range_low_var, bs_range_high_var, bs_order_var, ap_range_low_var, ap_range_high_var, ap_order_var, mp_kernel_var, bs2_range_low_var, bs2_range_high_var, bs2_order_var, mpb_kernel_var
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
        bs2_order_var = int(bs2_order.get())
        bs2_range_low_var = float(bs2_range_low.get())
        bs2_range_high_var = float(bs2_range_high.get())
        bs_order_var = int(bs_order.get())
        ap_range_low_var = float(ap_range_low.get())
        ap_range_high_var = float(ap_range_high.get())
        ap_order_var = int(ap_order.get())
        mp_kernel_var = int(mp_kernel.get())
        mpb_kernel_var = int(mpb_kernel.get())



        time.sleep(0.001)

def menu():
    global lp_check_var, lp_range, lp_order, hp_check_var, hp_range, hp_order, bp_check_var, bp_range_low, bp_range_high, bp_order, bs_check_var, bs_range_low, bs_range_high, bs_order, bs2_check_var, bs2_range_low, bs2_range_high, bs2_order, ap_check_var, ap_range_low, ap_range_high, ap_order, mp_check_var, mp_kernel, mpb_check_var, mpb_kernel
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
    bs2_frame = Frame(master)
    bs2_frame.pack(side=TOP)
    ap_frame = Frame(master)
    ap_frame.pack(side=TOP)
    mp_frame = Frame(master)
    mp_frame.pack(side=TOP)
    mpb_frame = Frame(master)
    mpb_frame.pack(side=TOP)

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
    bp_range_low.insert(0, "2")
    bp_range_low.pack(side=LEFT)
    bp_range_high = Entry(bp_frame)
    bp_range_high.insert(0, "80")
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

    bs2_check_var = IntVar()
    Checkbutton(bs2_frame, text="Bandstop", variable=bs2_check_var).pack(side=LEFT)
    Label(bs2_frame, text="Range").pack(side=LEFT)
    bs2_range_low = Entry(bs2_frame)
    bs2_range_low.insert(0, "95")
    bs2_range_low.pack(side=LEFT)
    bs2_range_high = Entry(bs2_frame)
    bs2_range_high.insert(0, "105")
    bs2_range_high.pack(side=LEFT)
    Label(bs2_frame, text="Order").pack(side=LEFT)
    bs2_order = Entry(bs2_frame)
    bs2_order.insert(0, "2")
    bs2_order.pack(side=LEFT)

    ap_check_var = IntVar()
    Checkbutton(ap_frame, text="Alpha", variable=ap_check_var).pack(side=LEFT)
    Label(ap_frame, text="Range").pack(side=LEFT)
    ap_range_low = Entry(ap_frame)
    ap_range_low.insert(0, "8")
    ap_range_low.pack(side=LEFT)
    ap_range_high = Entry(ap_frame)
    ap_range_high.insert(0, "13")
    ap_range_high.pack(side=LEFT)
    Label(ap_frame, text="Order").pack(side=LEFT)
    ap_order = Entry(ap_frame)
    ap_order.insert(0, "2")
    ap_order.pack(side=LEFT)

    mp_check_var = IntVar()
    Checkbutton(mp_frame, text="Median", variable=mp_check_var).pack(side=LEFT)
    Label(mp_frame, text="Kernel").pack(side=LEFT)
    mp_kernel = Entry(mp_frame)
    mp_kernel.insert(0, "3")
    mp_kernel.pack(side=LEFT)

    mpb_check_var = IntVar()
    Checkbutton(mpb_frame, text="Median Baseline", variable=mpb_check_var).pack(side=LEFT)
    Label(mpb_frame, text="Kernel").pack(side=LEFT)
    mpb_kernel = Entry(mpb_frame)
    mpb_kernel.insert(0, "151")
    mpb_kernel.pack(side=LEFT)


    mainloop()

UDP_IP = "192.168.4.21" # The IP that is printed in the serial monitor from the ESP32
SHARED_UDP_PORT = 50000
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Internet  # UDP
sock.connect((UDP_IP, SHARED_UDP_PORT))

# Create the PyQtGraph application
app = QApplication([])
win = pg.GraphicsLayoutWidget(show=True)
show = [3]
plot=[]
orig_plot = []
baseLine_plot=[]
alpha_plot=[]
fft_plot=[]

for i in show:
    plot.append(win.addPlot(title=f'Filtered {i}'))
    orig_plot.append(win.addPlot(title=f'Raw {i}'))
    win.nextRow()
    alpha_plot.append(win.addPlot(title=f'Alpha {i}'))
    fft_plot.append(win.addPlot(title=f'FFT {i}'))
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

def recvall(sock, size):
  data = bytearray()
  while len(data) < size:
    packet = sock.recv(size - len(data))
    if not packet:
      raise RuntimeError("Connection closed")
    data.extend(packet)
  return data

def loop():
    global ydata,num_samples
    while True:
        # Receive UDP packet
        data = recvall(sock,4*9)
        # print(data.decode())
        value = struct.unpack('iiiiiiiii', data)

        for i,j in enumerate(show):
            ydata[i][:-1] = ydata[i][1:]
            ydata[i][-1] = value[j]

        num_samples += 1

def update_plot():
    for i in range(len(show)):
        orig_curve[i].setData(y=ydata[i])
        filtered_curve[i].setData(y=filtered[i])
        alpha_curve[i].setData(y=alpha[i])
        fft_curve[i].setData(y=yfft[i][0:600//2], x=xfft)


def filter():
    global rate, filtered, ydata_avg, alpha, yfft, xfft, mpb_kernel_var
    T = 1/rate
    xfft = fftfreq(600, T)[:600//2]
    while True:
        # if rate>80*2:


                
        ydata_avg = ydata - np.mean(ydata,axis=1).reshape(-1,1)
        filtered_ = ydata_avg
        
        if lp_check_var.get() == 1:
            lowpass = signal.butter(lp_order_var, lp_range_var, 'lp', fs=rate, output='sos')
            filtered_ = signal.sosfilt(lowpass, ydata_avg)
        if hp_check_var.get() == 1:
            highpass = signal.butter(hp_order_var, hp_range_var, 'hp', fs=rate, output='sos')
            filtered_ = signal.sosfilt(highpass, filtered_)
        if bp_check_var.get() == 1:
            # bandpass = signal.butter(bp_order_var, [bp_range_low_var,bp_range_high_var], 'bandpass', fs=rate, output='sos')
            # filtered_ = signal.sosfilt(bandpass, filtered_)
            bandpass = signal.cheby2(bp_order_var, 40, [bp_range_low_var,bp_range_high_var], 'bandpass', fs=rate, output='sos')
            filtered_ = signal.sosfiltfilt(bandpass, filtered_)
        if bs_check_var.get() == 1:
            bandstop = signal.butter(bs_order_var, [bs_range_low_var,bs_range_high_var], 'bandstop', fs=rate, output='sos')
            filtered_ = signal.sosfilt(bandstop, filtered_)
        if bs2_check_var.get() == 1:
            bandstop2 = signal.butter(bs2_order_var, [bs2_range_low_var,bs2_range_high_var], 'bandstop', fs=rate, output='sos')
            filtered_ = signal.sosfilt(bandstop2, filtered_)

        if mpb_check_var.get() == 1:
            baseline=np.zeros((len(show),ydata.shape[1]))
            for i in range(len(show)):
                if mpb_kernel_var%2==0:
                    mpb_kernel_var+=1
                baseline[i]=signal.medfilt(filtered_[i], kernel_size=mpb_kernel_var)
                filtered_ = filtered_ - baseline

        # filtered=np.zeros((len(show),ydata.shape[1]))
        if mp_check_var.get() == 1:
            for i in range(len(show)):
                filtered_[i]=signal.medfilt(filtered_[i], kernel_size=mp_kernel_var)

        filtered = filtered_[:,200:800]
        
        alphapass = signal.butter(ap_order_var, [ap_range_low_var,ap_range_high_var], 'bandpass', fs=rate, output='sos')
        alpha = signal.sosfilt(alphapass, filtered)

        yfft = np.abs(fft(filtered))
            
            

# Set up a timer to update the plot
timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(10*len(show))  # Update every 10 ms

if __name__ == "__main__":
    sock.send('Hello ESP32'.encode())
    t7 = threading.Thread(target=menu)
    t7.start()
    time.sleep(0.1)
    t8 = threading.Thread(target=update_menu)
    t8.start()
    time.sleep(0.1)
    t1 = threading.Thread(target=loop)
    t1.start()
    time.sleep(0.1)
    t3 = threading.Thread(target=filter)
    t3.start()
    time.sleep(0.1)
    t6 = threading.Thread(target=update_plot)
    t6.start()
    time.sleep(0.1)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QApplication.instance().exec_()