import os
import pickle
import numpy as np
from scipy.io.wavfile import read 
import time
import sys
from FeatureExtraction import ExtractFeature
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# import PyQt5 library
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5 import QtWidgets,QtGui, uic, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox

# import module
from adding_dataset_individual import AddingNewDataset
import MakeSVMModel as svm


#global variable

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() 
        uic.loadUi('gui_mfcc.ui', self)
        # buat di preprocessing
        self.btn_predict.clicked.connect(self.test_dataset_test)
        self.btn_open_file.clicked.connect(self.openFile)
        self.btn_clear.clicked.connect(self.clear)
        self.btn_exit.clicked.connect(self.close)

        self.btn_open_dataTrain.clicked.connect(self.openDataTrain)
        self.btn_input_data_train.clicked.connect(self.inputDataTrain)
        self.btn_train_model.clicked.connect(self.trainModel)
        self.btn_clear_train.clicked.connect(self.clearTrain)

    #training function

    def openDataTrain(self):
        #open file dialog
        global dataTrain_location
        dataTrain_location, dataType = QFileDialog.getOpenFileName(self, 'Open Data Train','c\\', 'Wav Files (*.wav)')

        self.show_data_train_directory.setText(str(dataTrain_location))
        print(dataTrain_location)

    def inputDataTrain(self):
        label = self.inp_data_label.text()
        print(label)
        print(dataTrain_location)
        cd = AddingNewDataset(dataTrain_location,label)
        cd.createDataset()
        self.show_train_result.setText('Input Data ' + '"' + str(label) + '"' + ' Berhasil!') 
 

    def trainModel(self):
        #train model
        cr, svc = svm.make_model("dataset/dataset_train.csv")
        self.show_train_result.setText('---------------------- Training SVM Model ----------------------')
        self.show_train_result.append(str(cr))
        self.show_train_result.append('---------------------------------------------------------------')
        self.show_train_result.append(str(svc))
        self.show_train_result.append('---------------------------------------------------------------')
        self.show_train_result.append('Train Model Successfull')

        print('train successfull')

    def clearTrain(self):
        self.show_train_result.setText('Cleared!')
        self.show_data_train_directory.setText('')
        self.inp_data_label.setText('')


    #predict function

    def openFile(self):
        global dataset_location

        dataset_location, dataType = QFileDialog.getOpenFileName(self, 'Open Dataset','c\\', 'CSV files (*.csv)')        
        
        
        dataset_location = os.path.basename(dataset_location)
        dataset_location = "dataset/"+dataset_location
        print(dataset_location)

        self.show_dataset.setText('Detail Dataset')
        self.show_dataset.append('Dataset Location = ' + dataset_location)



    def testPredict(self, audio_path,speaker_name):
        '''
        @:param audio_path : Lokasi audio yang mau di prediksi (Audio harus berupa .wav)
        @:param speaker_name : Nama Pembicara
        @:return: hasil Pembiacara yang terdeteksi
        '''
        
        model = "dataset/model.pickle"
        load_model = pickle.load(open(model, 'rb'))

        # feature extraction dari audio_path 
        ef = ExtractFeature()
        feature = ef.extract_feature(audio_path)
        data = pd.DataFrame([feature])
        name = pd.DataFrame([speaker_name])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        pca = PCA().fit(X_scaled)
        X_pca = pca.transform(X_scaled)

        prediction = load_model.predict(data)
        print(prediction)
        print(classification_report(name,prediction))

    def test_dataset_test(self, dataset_location):
        '''
            @:param audio_path : Lokasi audio yang mau di prediksi (Audio harus berupa .wav)
            @:param speaker_name : Nama Pembicara
            @:retu
        '''
        #self.show_single_predict.setText(str(dataset_location))
        dataset_location = 'dataset/dataset_test.csv'
        model_location = 'dataset/model.pickle'
        load_model = pickle.load(open(model_location, 'rb'))
        dataset = pd.read_csv(dataset_location)
        X = dataset.drop('speaker',axis=1)
        Y = dataset['speaker']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X)
        pca = PCA().fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        prediction = load_model.predict(X_pca)
        print(classification_report(Y,prediction))
        # heat_map = sns.heatmap(confusion_matrix(Y,prediction))
        print(confusion_matrix(Y,prediction))
        plt.show()

        self.show_result.setText('---------------------- SVM Model Predict Result ----------------------')
        self.show_result.append('----------------------- Dataset  -----------------------')
        self.show_result.append('------------------------ Classification Report -------------------------')
        self.show_result.append(str(classification_report(Y,prediction)))
        self.show_result.append('------------------------ Confusion Matrix -------------------------')
        self.show_result.append(str(confusion_matrix(Y,prediction)))

    def clear(self):
        self.show_result.setText('Cleared!')
        self.show_dataset.setText('')
        self.show_single_predict.setText('')


    def run():
        # testPredict('datatest/airandwaters/4970-29095-0021.wav', 'airandwaters')
        #test_dataset_test('dataset/dataset_test.csv','dataset/model.pickle')
        print('testing')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = Ui()
    mainWin.show()
    sys.exit( app.exec_() )