'''
working criteria is as:
Three modules to be build:
    one for loading the training dataset
    another for loading testing dataset
    another for loading dataset for  accuracy testing
'''
import glob,os 
import pandas as pd 

class GetFiles:
    def __init__(self,dataset_path):

        '''
            dataset_path tu lokasi root dari datasetnya
        '''

        self.dataset_path = dataset_path

    def getTrainFiles(self,speaker_folder,types):
        '''
            type itu pake buat sub folder dari dataset nya atau
            bisa dibilang buat tipe modelnya, bisa train, test, atau predict
        '''
        dataframe_row = [] 
        #  tiap row di tatafrane_row ni tempat naruh audio_path dan targetnya (siapa yg ngomong)
        
        sub_files = os.listdir(self.dataset_path+"/"+types+"/"+speaker_folder)
        for files in sub_files:
            path_to_audio = self.dataset_path+"/"+types+"/"+speaker_folder+"/"+files

            # membuat row untuk dimasukin ke frame 
            dataframe_row.append([path_to_audio,speaker_folder])
        
        dataframe = pd.DataFrame(dataframe_row,columns=['audio_path','target_speaker'])

        return dataframe

    def getTestFiles(self):
        dataframe_row = []

        dataframe = pd.DataFrame()

        types = 'test'
        speaker_audio_folder = os.listdir(self.dataset_path+"/"+types)

        for folder in speaker_audio_folder:
            audio_files = os.listdir(self.dataset_path+"/"+types+"/"+folder)

            for files in audio_files:
                path_to_audio = self.dataset_path+"/"+types+folder+"/"+files
                dataframe_row.append([path_to_audio,folder])

            dataframe = pd.DataFrame(dataframe_row,columns=['audio_path','speaker'])
        return dataframe



