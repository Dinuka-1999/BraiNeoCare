from time import perf_counter
import csv
import numpy as np
import time
import threading
import pyqtgraph as pg
import socket
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtCore, QtWidgets
import struct
from scipy import signal
import os
import multiprocessing as mp
import subprocess

global RATE, record_status_mp, get_data_status, data_ch8_mp, data_ch8_filtered_mp, data_ch12_mp, data_ch12_filtered_mp, new_sample_count_mp, lock

RATE = 250
CHANNEL8 = ['T4-Cz','Fp2-Cz','C4-Cz','O2-Cz','O1-Cz','Fp1-Cz','C3-Cz','T3-Cz']
# CHANNEL8 = [1,2,3,4,5,6,7,8]
CHANNEL12 = ['Fp1-T3','T3-O1', 
            'Fp1-C3', 'C3-O1', 
            'Fp2-C4', 'C4-O2', 
            'Fp2-T4', 'T4-O2',
            'T3-C3', 'C3-Cz', 'Cz-C4', 'C4-T4']
TIME_LENGTH = 4
RATE = 250
NUM_SAMPLES = TIME_LENGTH*RATE

record_status_mp = mp.Value('b', False)
get_data_status_mp = mp.Value('b', False)
main_running_mp = mp.Value('b', True)
data_ch8_mp = mp.Array('f', NUM_SAMPLES*len(CHANNEL8))
data_ch8_filtered_mp = mp.Array('f', NUM_SAMPLES*len(CHANNEL8))
data_ch12_filtered_mp = mp.Array('f', NUM_SAMPLES*len(CHANNEL12))
data_acc_gyr_mp = mp.Array('f', NUM_SAMPLES*6)
new_sample_count_mp = mp.Value('i', 0)
ads_state_mp = mp.Value('i', 0)
filter_mp = mp.Value('b', False)
lock = mp.Lock()

def filter_data(get_data_status_mp, data_ch8_mp, data_ch8_filtered_mp, data_ch12_filtered_mp, new_sample_count_mp, main_running_mp, record_status_mp, filter_mp, data_acc_gyr_mp, lock):
    global CHANNEL8, CHANNEL12, NUM_SAMPLES, RATE
    first_time_record = 1

    b1, a1 = signal.cheby2(4, 20, [2,80], btype ='bandpass', fs=RATE)
    b2, a2 = signal.cheby2(2, 20, [48,52], btype ='bandstop', fs=RATE)
    b3, a3 = signal.cheby2(2, 20, [98,102], btype ='bandstop', fs=RATE)
    a = np.convolve(a1, a2)
    a = np.convolve(a, a3)
    b = np.convolve(b1, b2)
    b = np.convolve(b, b3)
    
    z8 = np.zeros((len(CHANNEL8), max(len(a), len(b))-1))
    
    data_ch8 = np.zeros((len(CHANNEL8),NUM_SAMPLES))
    data_ch8_filtered = np.zeros((len(CHANNEL8),NUM_SAMPLES))
    data_ch12_filtered = np.zeros((len(CHANNEL12),NUM_SAMPLES))

    start_time = 0

    while main_running_mp.value:
        if get_data_status_mp.value:
            with lock:
                data_ch8 = np.array(data_ch8_mp).reshape((len(CHANNEL8),-1))
                data_acc_gyr = np.array(data_acc_gyr_mp).reshape((6,-1))
                new_sample_count = new_sample_count_mp.value
                if new_sample_count!=0:
                    new_sample_count_mp.value = 0
                else:
                    continue
            # elapsed_time = time.perf_counter() - start_time
            # start_time = time.perf_counter()
            # rate = new_sample_count/elapsed_time
            # if rate<240:
            #     print('low', rate)
            # if rate>260:
            #     print('high', rate)
            
            data_ch8_filtered[:,:-new_sample_count] = data_ch8_filtered[:,new_sample_count:]
            if filter_mp.value:
                data_ch8_filtered[:,-new_sample_count:], z8 = signal.lfilter(b, a, data_ch8[:,-new_sample_count:], zi=z8)
            else:
                data_ch8_filtered[:,-new_sample_count:] = data_ch8[:,-new_sample_count:]
            
            with lock:
                data_ch8_filtered_mp[:] = data_ch8_filtered.reshape(-1)
            # data_ch8_filtered[:,-new_sample_count:] = filtered_data[:,0]

            data_ch12_filtered[:,:-new_sample_count] = data_ch12_filtered[:,new_sample_count:]
            data_ch12_filtered[:,-new_sample_count:] = [data_ch8_filtered[5,-new_sample_count:]-data_ch8_filtered[7,-new_sample_count:],
                                                        data_ch8_filtered[7,-new_sample_count:]-data_ch8_filtered[4,-new_sample_count:],
                                                        data_ch8_filtered[5,-new_sample_count:]-data_ch8_filtered[6,-new_sample_count:],
                                                        data_ch8_filtered[6,-new_sample_count:]-data_ch8_filtered[4,-new_sample_count:],
                                                        data_ch8_filtered[1,-new_sample_count:]-data_ch8_filtered[2,-new_sample_count:],
                                                        data_ch8_filtered[2,-new_sample_count:]-data_ch8_filtered[3,-new_sample_count:],
                                                        data_ch8_filtered[1,-new_sample_count:]-data_ch8_filtered[0,-new_sample_count:],
                                                        data_ch8_filtered[0,-new_sample_count:]-data_ch8_filtered[3,-new_sample_count:],
                                                        data_ch8_filtered[7,-new_sample_count:]-data_ch8_filtered[6,-new_sample_count:],
                                                        data_ch8_filtered[6,-new_sample_count:],
                                                        -data_ch8_filtered[2,-new_sample_count:],
                                                        data_ch8_filtered[2,-new_sample_count:]-data_ch8_filtered[0,-new_sample_count:]]
                                                        
            with lock:
                data_ch12_filtered_mp[:] = data_ch12_filtered.reshape(-1)
            time.sleep(.1)
            if record_status_mp.value:
                if first_time_record:
                    first_time_record = 0
                    # create a folder for recordings if not exist
                    if not os.path.exists('recordings'):
                        os.makedirs('recordings')
                    csvfile = open(f'recordings/Recording-{time.strftime("%Y%m%d-%H%M%S")}.csv', 'w', newline='')
                    writer = csv.writer(csvfile)
                    time_array = np.zeros(len(CHANNEL12)+len(CHANNEL8)+6)
                    time_array[0] = time.time()
                    writer.writerow(time_array)
                    for i in range(0,NUM_SAMPLES-1):
                        # writer.writerow(data_ch12_filtered[:,i])
                        writer.writerow(np.hstack((data_ch12_filtered[:,i], data_ch8[:,i], data_acc_gyr[:,i])))

                for i in range(new_sample_count):
                    # writer.writerow(data_ch12_filtered[:,-new_sample_count+i])
                    writer.writerow(np.hstack((data_ch12_filtered[:,-new_sample_count+i], data_ch8[:,-new_sample_count+i], data_acc_gyr[:,-new_sample_count+i])))
            else:
                if first_time_record==0:
                    time_array = np.zeros(len(CHANNEL12)+len(CHANNEL8)+6)
                    time_array[0] = time.time()
                    writer.writerow(time_array)
                    first_time_record = 1
                    csvfile.close()

                

        else:
            z8 = np.zeros((len(CHANNEL8), max(len(a), len(b))-1))

            data_ch8 = np.zeros((len(CHANNEL8),NUM_SAMPLES))
            data_ch8_filtered = np.zeros((len(CHANNEL8),NUM_SAMPLES))
            data_ch12_filtered = np.zeros((len(CHANNEL12),NUM_SAMPLES))

            if first_time_record==0:
                first_time_record = 1
                csvfile.close()

            time.sleep(0.2)

def get_data(get_data_status_mp, data_ch8_mp, new_sample_count_mp, main_running_mp, ads_state_mp, data_acc_gyr_mp , lock):
    
    global CHANNEL8, NUM_SAMPLES, RATE

    first_time = 1
    TCP_IP = "192.168.4.21" # The IP that is printed in the serial monitor from the ESP32
    SHARED_TCP_PORT = 50000
    
    def recvall(sock, size):
        data = bytearray()
        while len(data) < size:
            try:
                packet = sock.recv(size - len(data))
                data.extend(packet)
            except:
                get_data_status_mp.value = False
                data = 'ca'
                break
        return data
    
    def init_ads(sock, state):
        sock.connect((TCP_IP, SHARED_TCP_PORT))
        mystr = ''
        if state.value==0:
            mystr+= str(int('11101100',2))+'\n' # CONFIG3
            mystr+= str(int('10010110',2))+'\n' # CONFIG1
            mystr+= str(int('11010000',2))+'\n' # CONFIG2
            mystr+= str(int('01100000',2))+'\n' # CH1SET
            mystr+= str(int('01100000',2))+'\n' # CH2SET
            mystr+= str(int('01100000',2))+'\n' # CH3SET
            mystr+= str(int('01100000',2))+'\n' # CH4SET
            mystr+= str(int('01100000',2))+'\n' # CH5SET
            mystr+= str(int('01100000',2))+'\n' # CH6SET
            mystr+= str(int('01100000',2))+'\n' # CH7SET
            mystr+= str(int('01100000',2))+'\n' # CH8SET
            mystr+= str(int('00000000',2))+'\n' # BIAS_SENSP
            mystr+= str(int('00000000',2))+'\n' # BIAS_SENSN
            mystr+= str(int('00100000',2))+'\n' # MISC1
        else:
            mystr+= str(int('11101100',2))+'\n' # CONFIG3
            mystr+= str(int('10010110',2))+'\n' # CONFIG1
            mystr+= str(int('11010100',2))+'\n' # CONFIG2
            mystr+= str(int('01100000',2))+'\n' # CH1SET
            mystr+= str(int('01100000',2))+'\n' # CH2SET
            mystr+= str(int('01100000',2))+'\n' # CH3SET
            mystr+= str(int('01100000',2))+'\n' # CH4SET
            mystr+= str(int('01100000',2))+'\n' # CH5SET
            mystr+= str(int('01100000',2))+'\n' # CH6SET
            mystr+= str(int('01100000',2))+'\n' # CH7SET
            mystr+= str(int('01100000',2))+'\n' # CH8SET
            mystr+= str(int('11111111',2))+'\n' # BIAS_SENSP
            mystr+= str(int('11111111',2))+'\n' # BIAS_SENSN
            mystr+= str(int('00100000',2))+'\n' # MISC1
        sock.sendall(mystr.encode())

    while main_running_mp.value:

        if get_data_status_mp.value:
            if first_time:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Internet  # TCP
                first_time = 0
                init_ads(sock, ads_state_mp)
            #start a thread to find rate
            
            # Receive TCP packet
            latest_data = recvall(sock,4*17)
            if latest_data == 'ca':
                value = np.zeros(9)
            else:
                unpacked = struct.unpack('iiiiiiiiifffffffi', latest_data)
                value = np.array(unpacked[0:9])
                acc_gyr = np.array(unpacked[9:15])
                temp = unpacked[15]
                bat = unpacked[16]
            value = value*((2*5/24)/(2**24))*1000 # page 38 ads1299 datasheet
            with lock:
                data_ch8_mp[:-1] = data_ch8_mp[1:]
                data_acc_gyr_mp[:-1] = data_acc_gyr_mp[1:]
                for i in range(1,len(CHANNEL8)+1):
                    data_ch8_mp[i*NUM_SAMPLES-1] = value[i]
                for i in range(6):
                    data_acc_gyr_mp[i*NUM_SAMPLES-1] = acc_gyr[i]
                new_sample_count_mp.value += 1
            
            # with lock:
            #     data_ch8_mp[:-1] = data_ch8_mp[1:]
            #     for i in range(len(CHANNEL8)):
            #         data_ch8_mp[i*NUM_SAMPLES-1] = np.random.rand(1)
            #     new_sample_count_mp.value += 1
            # time.sleep(0.003)
        else:
            if first_time == 0:
                first_time = 1
                sock.close()
            time.sleep(0.2)
       
class BraiNeoCareGUI(QtWidgets.QWidget):
    def __init__(self):
        
        global CHANNEL8, CHANNEL12, NUM_SAMPLES, RATE, record_status_mp, get_data_status_mp, data_ch8_filtered_mp, data_ch12_filtered_mp, new_sample_count_mp, filter_mp, lock
        self.time_array = np.arange(0, NUM_SAMPLES/RATE, 1/RATE)
        QtWidgets.QWidget.__init__(self)
        self.setupGUI()
        self.setupTree()

    def setupGUI(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.layout.addWidget(self.splitter)
        
        self.tree = ParameterTree(showHeader=False)
        self.splitter.addWidget(self.tree)
        
        self.splitter2 = QtWidgets.QSplitter()
        self.splitter2.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.splitter.addWidget(self.splitter2)
        
        self.main_plots = pg.GraphicsLayoutWidget()
        self.splitter2.addWidget(self.main_plots)

        # self.main_plots.setMinimumWidth(2000)
        # self.main_plots.setMinimumHeight(2000)
        # self.scroll = pg.QtWidgets.QScrollArea()
        # self.scroll.setWidget(self.main_plots)
        # self.scroll.setWidgetResizable(True)
        # self.splitter2.addWidget(self.scroll)

    def setupTree(self):
        self.params = Parameter.create(name='params', type='group', children=[
            dict(name='Check Connectivity', type='action'),
            dict(name='Stop', type='action', enabled=False),
            dict(name='Live Run', type='group', children=[
                dict(name='Start', type='action'),
                dict(name='Record', type='action', enabled=False),
                dict(name='Stop Recording', type='action', enabled=False),
                dict(name='IMU', type='action'),
            ]),
            dict(name='Load', type='action'),
            dict(name='Filter', type='bool', value=False)
        ])
        self.tree.setParameters(self.params, showTop=False)
        self.params.param('Check Connectivity').sigActivated.connect(self.check_connectivity)
        self.params.param('Stop').sigActivated.connect(self.stop)
        self.params.param('Live Run', 'Start').sigActivated.connect(self.start_live_run)
        self.params.param('Live Run', 'Record').sigActivated.connect(self.record)
        self.params.param('Live Run', 'Stop Recording').sigActivated.connect(self.stop_record)
        self.params.param('Live Run', 'IMU').sigActivated.connect(self.imu)
        self.params.param('Load').sigActivated.connect(self.load)
        self.params.param('Filter').sigValueChanged.connect(self.filter)

    def check_connectivity(self):
        if "BraiNeoCare" not in (subprocess.check_output("netsh wlan show interfaces")).decode('utf-8'):
            QtWidgets.QMessageBox.information(self, "WiFi Error", "Please connect your WiFi to 'BraiNeoCare' and try again.")
        else:
            self.stop()
            self.params.param('Stop').setOpts(enabled=True)
            self.params.param('Check Connectivity').setOpts(enabled=False)
            self.params.param('Load').setOpts(enabled=False)
            self.plots=[]
            c=0
            for i in CHANNEL8:
                self.plots.append(self.main_plots.addPlot(title=f'Channel {i}'))
                c+=1
                if c%2==0:
                    self.main_plots.nextRow()
            self.curves=[]
            for i in range(len(CHANNEL8)):
                if i!=0:
                    self.plots[i].setXLink(self.plots[0])
                    # self.plots[i].setYLink(self.plots[0])
                self.curves.append(self.plots[i].plot())


            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_plot_connectivity)
            self.timer.start(100) 
            with lock:
                get_data_status_mp.value = True
                ads_state_mp.value = 0

    def start_live_run(self):
        if "BraiNeoCare" not in (subprocess.check_output("netsh wlan show interfaces")).decode('utf-8'):
            QtWidgets.QMessageBox.information(self, "WiFi Error", "Please connect your WiFi to 'BraiNeoCare' and try again.")
        else:
            self.stop()
            self.params.param('Live Run', 'Record').setOpts(enabled=True)
            self.params.param('Live Run', 'Start').setOpts(enabled=False)
            self.params.param('Stop').setOpts(enabled=True)
            self.params.param('Load').setOpts(enabled=False)
            self.plots=[]
            c = 0
            for i in CHANNEL12:
                self.plots.append(self.main_plots.addPlot(title=i))
                c+=1
                if c%2==0:
                    self.main_plots.nextRow()
            self.curves=[]
            for i in range(len(CHANNEL12)):
                # self.plots[i].setMinimumHeight(100)
                if i!=0:
                    self.plots[i].setXLink(self.plots[0])
                    # self.plots[i].setYLink(self.plots[0])
                    # title on the left
                    # self.plots[i].setLabel('left', 'Channel '+str(i), units='V')
                self.curves.append(self.plots[i].plot())


            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_plot_live_run)
            self.timer.start(100) 

            with lock:
                ads_state_mp.value = 1
                get_data_status_mp.value = True

    def imu(self):
        if "BraiNeoCare" not in (subprocess.check_output("netsh wlan show interfaces")).decode('utf-8'):
            QtWidgets.QMessageBox.information(self, "WiFi Error", "Please connect your WiFi to 'BraiNeoCare' and try again.")
        else:
            self.stop()
            self.params.param('Live Run', 'Record').setOpts(enabled=True)
            self.params.param('Live Run', 'Start').setOpts(enabled=False)
            self.params.param('Stop').setOpts(enabled=True)
            self.params.param('Load').setOpts(enabled=False)
            self.plots=[]
            c = 0
            for i in range(6):
                self.plots.append(self.main_plots.addPlot(title=f'IMU {i}'))
                c+=1
                if c%2==0:
                    self.main_plots.nextRow()
            self.curves=[]
            for i in range(6):
                self.plots[i].setXLink(self.plots[0])
                self.curves.append(self.plots[i].plot())

            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_plot_imu)
            self.timer.start(100) 
            with lock:
                get_data_status_mp.value = True
                ads_state_mp.value = True

    def stop(self):
        self.main_plots.clear()

        if hasattr(self, 'timer'):
            self.timer.stop()

        self.data_ch8_filtered = np.zeros((len(CHANNEL8),NUM_SAMPLES))
        self.data_ch12_filtered = np.zeros((len(CHANNEL12),NUM_SAMPLES))
        with lock:
            get_data_status_mp.value = False
            data_ch8_filtered_mp[:] = self.data_ch8_filtered.reshape(-1)
            data_ch8_mp[:] = self.data_ch8_filtered.reshape(-1)
            data_ch12_filtered_mp[:] = self.data_ch12_filtered.reshape(-1)
            new_sample_count_mp.value = 0

        with lock:
            record_status_mp.value = False

        self.params.param('Stop').setOpts(enabled=False)
        self.params.param('Check Connectivity').setOpts(enabled=True)
        self.params.param('Live Run', 'Start').setOpts(enabled=True)
        self.params.param('Live Run', 'Record').setOpts(enabled=False)
        self.params.param('Live Run', 'Stop Recording').setOpts(enabled=False)
        self.params.param('Load').setOpts(enabled=True)
        
    def record(self):
        with lock:
            record_status_mp.value = True
        self.params.param('Live Run', 'Record').setOpts(enabled=False)
        self.params.param('Live Run', 'Stop Recording').setOpts(enabled=True)

    def stop_record(self):
        with lock:
            record_status_mp.value = False
        self.params.param('Live Run', 'Record').setOpts(enabled=True)
        self.params.param('Live Run', 'Stop Recording').setOpts(enabled=False)

    def load(self):
        #browse for file
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Data..", "", "CSV Files (*.csv)")
        if not filename:
            return
        # load csv file as numpy array
        data = np.genfromtxt(filename, delimiter=',')
        start_time = data[0,0]
        # print(data.shape)
        stop_time = data[-1,0]
        time_length = stop_time-start_time
        data = data[1:-1,:]
        # plot data
        self.main_plots.clear()
        self.plots=[]
        c = 0
        for i in CHANNEL12:
            self.plots.append(self.main_plots.addPlot(title=f'Channel {i}', axisItems = {'bottom': pg.AxisItem('bottom', units='s')}))
            c+=1
            if c%2==0:
                self.main_plots.nextRow()
        for i in CHANNEL8:
            self.plots.append(self.main_plots.addPlot(title=f'Channel {i}', axisItems = {'bottom': pg.AxisItem('bottom', units='s')}))
            c+=1
            if c%2==0:
                self.main_plots.nextRow()
        for i in range(6):
            self.plots.append(self.main_plots.addPlot(title=f'IMU {i}', axisItems = {'bottom': pg.AxisItem('bottom', units='s')}))
            c+=1
            if c%2==0:
                self.main_plots.nextRow()
        self.curves=[]
        for i in range(len(CHANNEL12)):
            self.plots[i].setXLink(self.plots[0])
            self.curves.append(self.plots[i].plot())
        for i in range(len(CHANNEL8)):
            self.plots[i+len(CHANNEL12)].setXLink(self.plots[0])
            self.curves.append(self.plots[i+len(CHANNEL12)].plot())
        for i in range(6):
            self.plots[i+len(CHANNEL12)+len(CHANNEL8)].setXLink(self.plots[0])
            # self.plots[i+len(CHANNEL12)+len(CHANNEL8)].setYLink(self.plots[0])
            self.curves.append(self.plots[i+len(CHANNEL12)+len(CHANNEL8)].plot())
        # time_array = np.arange(0, data.shape[0]/RATE, 1/RATE)
        time_array = np.linspace(0, time_length, data.shape[0])
        for i in range(len(CHANNEL12)):
            self.curves[i].setData(x=time_array, y=data[:,i])
        for i in range(len(CHANNEL8)):
            self.curves[i+len(CHANNEL12)].setData(x=time_array, y=data[:,i+len(CHANNEL12)])
        for i in range(6):
            self.curves[i+len(CHANNEL12)+len(CHANNEL8)].setData(x=time_array, y=data[:,i+len(CHANNEL12)+len(CHANNEL8)])
            

    def update_plot_connectivity(self):
        # with lock:
        self.data_ch8_filtered = np.array(data_ch8_filtered_mp).reshape((len(CHANNEL8),-1))
        for i in range(len(CHANNEL8)):
            #change curve color if variance is lower than 0.1
            if np.var(self.data_ch8_filtered[i][int(-NUM_SAMPLES/50):])<0.1:
                self.curves[i].setPen('g')
            else:
                self.curves[i].setPen('r')
            self.curves[i].setData(x=self.time_array, y=self.data_ch8_filtered[i])

    def update_plot_live_run(self):
        # with lock:
        self.data_ch12_filtered = np.array(data_ch12_filtered_mp).reshape((len(CHANNEL12),-1))
        for i in range(len(CHANNEL12)):
            self.curves[i].setData(x=self.time_array, y=self.data_ch12_filtered[i])
        
    def update_plot_imu(self):
        # with lock:
        self.data_acc_gyr = np.array(data_acc_gyr_mp).reshape((6,-1))
        for i in range(6):
            self.curves[i].setData(x=self.time_array, y=self.data_acc_gyr[i])

    def closeEvent(self, event):
        main_running_mp.value = False

    def filter(self):
        filter_mp.value = self.params.param('Filter').value()

        
if __name__ == '__main__':
    
    get_data_process = mp.Process(target=get_data, args=(get_data_status_mp, data_ch8_mp, new_sample_count_mp, main_running_mp, ads_state_mp, data_acc_gyr_mp, lock))
    filter_data_process = mp.Process(target=filter_data, args=(get_data_status_mp, data_ch8_mp, data_ch8_filtered_mp, data_ch12_filtered_mp, new_sample_count_mp, main_running_mp, record_status_mp, filter_mp, data_acc_gyr_mp, lock))
    get_data_process.start()
    filter_data_process.start()
    app = pg.mkQApp()
    win = BraiNeoCareGUI()
    win.setWindowTitle("BraiNeoCare")
    win.showMaximized()
    # win.resize(1100,700)
    pg.exec()
    get_data_process.join()
    filter_data_process.join()