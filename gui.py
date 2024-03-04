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
import sys
import tensorflow as tf
import Read_Data as rd
import cv2
import tensorflow_addons as tfa

class BraiNeoCareGUI(QtWidgets.QWidget):
    def __init__(self):
        self.CHANNEL9 = [1,2,3,4,5,6,7,8,9]
        self.CHANNEL12 = ['Fp1-T3', 'Fp1-C3', 'Fp2-C4', 'Fp2-T4', 'T3-C3', 'C3-Cz', 'Cz-C4', 'C4-T4', 'T3-O1', 'C3-O1', 'C4-O2', 'T4-O2']
        self.mean_std=np.load('mean_std.npy')
        self.mean=self.mean_std[0]
        self.std=self.mean_std[1]
        self.TIME_LENGTH = 12
        self.RATE = 256
        self.NUM_SAMPLES = self.TIME_LENGTH*self.RATE
        self.record_status = False
        self.get_data_status = False
        self.data_ch9 = np.zeros((len(self.CHANNEL9),self.NUM_SAMPLES))
        self.data_ch9_filtered = np.zeros((len(self.CHANNEL9),self.NUM_SAMPLES))
        self.data_ch12 = np.zeros((len(self.CHANNEL12),self.NUM_SAMPLES))
        self.data_ch12_filtered = np.zeros((len(self.CHANNEL12),self.NUM_SAMPLES))
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
            dict(name='Clear', type='action'),
            dict(name='Live Run', type='group', children=[
                dict(name='Start', type='action'),
                dict(name='Stop', type='action'),
                dict(name='Detect', type='action'),
                dict(name='Record', type='action'),
                dict(name='Stop Recording', type='action')
            ]),
            dict(name='Load', type='action')
        ])
        self.tree.setParameters(self.params, showTop=False)
        self.params.param('Check Connectivity').sigActivated.connect(self.check_connectivity)
        self.params.param('Clear').sigActivated.connect(self.clear)
        self.params.param('Live Run', 'Start').sigActivated.connect(self.start_live_run)
        self.params.param('Live Run', 'Stop').sigActivated.connect(self.clear)
        self.params.param('Live Run', 'Detect').sigActivated.connect(self.detect)
        self.params.param('Live Run', 'Record').sigActivated.connect(self.record)
        self.params.param('Live Run', 'Stop Recording').sigActivated.connect(self.stop_record)
        self.params.param('Load').sigActivated.connect(self.load)

    def check_connectivity(self):
        self.clear()
        self.plots=[]
        for i in self.CHANNEL9:
            self.plots.append(self.main_plots.addPlot(title=f'Channel {i}'))
            self.main_plots.nextRow()
        self.curves=[]
        for i in range(len(self.CHANNEL9)):
            self.curves.append(self.plots[i].plot())

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot_connectivity)
        self.timer.start(100)  # Update every 100 ms

        self.get_data_status = True
        self.get_data_thread = threading.Thread(target=self.get_data)
        self.get_data_thread.start()

    def clear(self):
        self.main_plots.clear()

        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'get_data_thread'):
            self.get_data_status = False
            self.get_data_thread.join()

        self.data_ch9 = np.zeros((len(self.CHANNEL9),self.NUM_SAMPLES))
        self.data_ch9_filtered = np.zeros((len(self.CHANNEL9),self.NUM_SAMPLES))
        self.data_ch12 = np.zeros((len(self.CHANNEL12),self.NUM_SAMPLES))
        self.data_ch12_filtered = np.zeros((len(self.CHANNEL12),self.NUM_SAMPLES))

    def start_live_run(self):
        self.clear()
        self.plots=[]
        c = 0
        for i in self.CHANNEL12:
            self.plots.append(self.main_plots.addPlot(title=f'Channel {i}'))
            c+=1
            if c%2==0:
                self.main_plots.nextRow()
        self.curves=[]
        for i in range(len(self.CHANNEL12)):
            # self.plots[i].setMinimumHeight(100)
            self.curves.append(self.plots[i].plot())

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot_live_run)
        self.timer.start(100)  # Update every 100 ms

        self.get_data_status = True
        self.get_data_thread = threading.Thread(target=self.get_data)
        self.get_data_thread.start()

    def detect(self):
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.start()

    def inference_loop(self):
        while self.get_data_status:
            self.downsampled_signal = signal.resample(self.data_ch12_filtered, 384,axis=1)
            self.PreprocesSignal(self.downsampled_signal)
            self.explainability(self.exp_model, self.processed_signal)
            self.update_gui_with_inference_results(self.prediction, self.heatmap)
            time.sleep(1/self.RATE)

    def update_gui_with_inference_results(self, prediction, heatmap):
        # print(1 if prediction>0.5 else 0)
        print(prediction)
        self.heatmap1 = cv2.resize(heatmap, (384,12), interpolation=cv2.INTER_LINEAR)

    def record(self):
        #write data to csv file
        if not os.path.exists('recordings'):
            os.makedirs('recordings')
        self.csvfile = open(f'recordings/Recording-{time.strftime("%Y%m%d-%H%M%S")}.csv', 'w', newline='')
        self.writer = csv.writer(self.csvfile)
        self.record_status = True

    def stop_record(self):
        self.record_status = False
        self.csvfile.close()

    def load(self):
        #browse for file
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Data..", "", "CSV Files (*.csv)")
        if not filename:
            return
        # load csv file as numpy array
        data = np.genfromtxt(filename, delimiter=',')
        # plot data
        self.main_plots.clear()
        self.plots=[]
        c = 0
        for i in self.CHANNEL12:
            self.plots.append(self.main_plots.addPlot(title=f'Channel {i}'))
            c+=1
            if c%2==0:
                self.main_plots.nextRow()
        self.curves=[]
        for i in range(len(self.CHANNEL12)):
            self.plots[i].setXLink(self.plots[0])
            self.curves.append(self.plots[i].plot())

        for i in range(len(self.CHANNEL12)):
            self.curves[i].setData(data[:,i])

    def update_plot_connectivity(self):
        for i in range(len(self.CHANNEL9)):
            self.curves[i].setData(self.data_ch9_filtered[i])

    def update_plot_live_run(self):
        for i in range(len(self.CHANNEL12)):
            self.curves[i].setData(self.data_ch12_filtered[i])
        if self.record_status:
            self.writer.writerow(self.data_ch12_filtered[:,-1])
    
    def PreprocesSignal(self,signal1):
        self.signal2=(signal1-self.mean)/self.std
        self.signal2=np.expand_dims(self.signal2,axis=-1)
        self.processed_signal=np.expand_dims(self.signal2,axis=0)
    
    def explainability(self,exp_model,EEG):
        with tf.GradientTape() as tape:
            GAT_outputs, predictions = exp_model(EEG)
            Class=predictions[:,0]

        grads = tape.gradient(Class, GAT_outputs)
        heatmap = GAT_outputs[0] *  tf.reduce_mean(grads, axis=(0,1,2))
        self.heatmap = (tf.nn.relu(heatmap)/tf.reduce_max(heatmap)).numpy()
        self.prediction=tf.nn.sigmoid(predictions[0][0]).numpy()

    def get_data(self):

        # self.detect()
        EEG,s_time,e_time = rd.read_a_file('./Datasets/zenodo_eeg/eeg1.edf',1,pre_processing=False)
        b, a = signal.cheby2(7, 20, [1,30], 'bandpass', fs=self.RATE)
        print(s_time,e_time)
        global z9,z12
        z9 = np.transpose(np.expand_dims(signal.lfilter_zi(b, a), axis=-1) * np.expand_dims(self.data_ch9[:,-1], axis=0))
        z12 = np.transpose(np.expand_dims(signal.lfilter_zi(b, a), axis=-1) * np.expand_dims(self.data_ch12[:,-1], axis=0))
        # time to run one iteration of the loop
        def filter_data():

            global z9,z12
            self.data_ch9_filtered[:,:-1] = self.data_ch9_filtered[:,1:]
            filtered_data, z9 = signal.lfilter(b, a, np.expand_dims(self.data_ch9[:,-1], axis=-1), zi=z9)
            self.data_ch9_filtered[:,-1] = filtered_data[:,0]

            self.data_ch12_filtered[:,:-1] = self.data_ch12_filtered[:,1:]
            filtered_data, z12 = signal.lfilter(b, a, np.expand_dims(self.data_ch12[:,-1], axis=-1), zi=z12)
            self.data_ch12_filtered[:,-1] = filtered_data[:,0]

        # def recvall(sock, size):
        #     data = bytearray()
        #     while len(data) < size:
        #         packet = sock.recv(size - len(data))
        #         if not packet:
        #             raise RuntimeError("Connection closed")
        #         data.extend(packet)
        #     return data
        
        # TCP_IP = "192.168.4.21" # The IP that is printed in the serial monitor from the ESP32
        # SHARED_TCP_PORT = 50000
        # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Internet  # TCP
        # sock.connect((TCP_IP, SHARED_TCP_PORT))

        # while self.get_data_status:
        #     # Receive TCP packet
        #     latest_data = recvall(sock,4*9)
        #     print(latest_data.decode())
        #     value = np.array(struct.unpack('iiiiiiiii', latest_data))
        #     value = value*((2*5/24)/(2**24))*1000 # page 38 ads1299 datasheet
        #     self.data_ch9[:,:-1] = self.data_ch9[:,1:]
        #     self.data_ch9[:,-1] = value
        

        # while self.get_data_status:
        #     self.data_ch9[:,:-1] = self.data_ch9[:,1:]
        #     self.data_ch9[:,-1] = np.random.rand(9)
            
        #     self.data_ch12[:,:-1] = self.data_ch12[:,1:]
        #     self.data_ch12[:,-1] = [self.data_ch9[1,-1]-self.data_ch9[2,-1],
        #                             self.data_ch9[1,-1]-self.data_ch9[3,-1],
        #                             self.data_ch9[1,-1]-self.data_ch9[4,-1],
        #                             self.data_ch9[1,-1]-self.data_ch9[5,-1],
        #                             self.data_ch9[1,-1]-self.data_ch9[6,-1],
        #                             self.data_ch9[1,-1]-self.data_ch9[7,-1],
        #                             self.data_ch9[2,-1]-self.data_ch9[3,-1],
        #                             self.data_ch9[2,-1]-self.data_ch9[4,-1],
        #                             self.data_ch9[2,-1]-self.data_ch9[5,-1],
        #                             self.data_ch9[2,-1]-self.data_ch9[6,-1],
        #                             self.data_ch9[2,-1]-self.data_ch9[7,-1],
        #                             self.data_ch9[3,-1]-self.data_ch9[4,-1]]
            
        #     threading.Thread(target=filter_data).start()
        #     time.sleep(1/self.RATE)
        l=0
        while self.get_data_status:
            s=time.time()
            np.roll(self.data_ch9, -1, axis=1)
            # self.data_ch9[:,:-1] = self.data_ch9[:,1:]
            self.data_ch9[:,-1] = EEG[0:9,l]
            
            np.roll(self.data_ch12, -1, axis=1)
            # self.data_ch12[:,:-1] = self.data_ch12[:,1:]
            self.data_ch12[:,-1] = EEG[:,l]
            # filter_data()
            threading.Thread(target=filter_data).start()
            l+=1
            if l>=len(EEG[0]):
                self.get_data_status = False
            # time.sleep(1/self.RATE)
            # e=time.time()
            # new_time = time.time()
            # print(e-s)
        

    def closeEvent(self, event):
        os._exit(0)

if __name__ == '__main__':
    model = tf.keras.models.load_model("GAT_correct_7/cp_23.ckpt")
    model.layers[-1].activation = None
    exp_model=tf.keras.Model(model.inputs,[model.get_layer("gat_layer_5").output,model.output])
    app = pg.mkQApp()
    win = BraiNeoCareGUI()
    win.exp_model = exp_model
    win.setWindowTitle("BraiNeoCare")
    win.showMaximized()
    # win.resize(1100,700)
    sys.exit(pg.exec())