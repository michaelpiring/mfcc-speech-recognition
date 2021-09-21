import pandas as pd
from FeatureExtraction import ExtractFeature
from GetFiles import GetFiles as gf 
import os
from pydub import AudioSegment
from pathlib import Path
import csv as writer
class AddingNewDataset:
    def __init__(self,file_location,name):
        '''
            param : 
            -- database itu nama tempat nyimpen file suaranya
        '''
        self.database = 'dataset/dataset_train.csv'
        self.file = file_location
        self.name = name

    def createDataset(self):

            # nyiapin label buat datasetnya
            label = list(range(210))
            label = map(str,label)
            label = list(label)
            label.append('speaker')
            data = []

            try :
                if self.file:
                    ef = ExtractFeature()
                    feature = ef.extract_feature(self.file)
                    feature = list(feature)
                    feature.append(self.name)
                    data.append(feature)
                    result = pd.DataFrame(data,columns=label)
                    result.to_csv(self.database,mode='a', header=False,index=False)                    
            except Exception as e:
                print(e)
        

    def getFolder(self):
        folder = os.listdir(self.database)

        return folder

    def getFiles(self,folder_name):
        files = os.listdir(self.database + "/" +folder_name)
        return files

    
#cd = AddingNewDataset("C:/Users/Microsoft/Documents/Michael/Kampus/semester 6/Teknologi Biometrika/voice_recognitiontes/voice_recognition/ngurah.wav",'ngurah')
#cd.createDataset()    
