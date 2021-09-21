import pandas as pd
from FeatureExtraction import ExtractFeature
from GetFiles import GetFiles as gf 
import os
from pydub import AudioSegment
from pathlib import Path

class CreateDataset:
    def __init__(self,database,name):
        '''
            param : 
            -- database itu nama tempat nyimpen file suaranya
        '''
        self.database = database
        self.name = name

    def createDataset(self):

        folders = self.getFolder()
        # nyiapin label buat datasetnya
        label = list(range(210))
        label = map(str,label)
        label = list(label)
        label.append('speaker')
        data = []

        try :
            for folder in folders:
                files = self.getFiles(folder)
                for file in files:
                    audio_path = self.database + "/" + folder + "/" + file

                    if audio_path.endswith('.wav'):
                        audio = audio_path
                        if audio:
                            ef = ExtractFeature()
                            feature = ef.extract_feature(audio)
                            feature = list(feature)
                            feature.append(folder)
                            data.append(feature)
                    elif audio_path.endswith('.txt'):
                        continue
                    else:
                        continue
            result = pd.DataFrame(data, columns=label)  
            result.to_csv('dataset/dataset_'+ self.name+ '.csv', index=False)
            return result                          
        except Exception as e:
            print(e)
        

    def getFolder(self):
        folder = os.listdir(self.database)

        return folder

    def getFiles(self,folder_name):
        files = os.listdir(self.database + "/" +folder_name)
        return files

    
cd = CreateDataset("datatrain",'train')
cd.createDataset()    
