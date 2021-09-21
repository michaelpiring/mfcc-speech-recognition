''' 
    File yang digunakan untuk melakukan feature extraction 
    dari voice dengan menggunakan MFCC (Mel-Frequency Cepstral Coefficient)
    
    Tahapan Feature extraction yaitu dapat dilihat pada link berikut.
    https://link.springer.com/content/pdf/bbm%3A978-3-319-49220-9%2F1.pdf
    
    ada dua liblary yang dapat melakukan MFCC yaitu librosa dan python_speech_features

'''

import numpy as np
from sklearn import preprocessing
import python_speech_features as speech_features
from scipy.io.wavfile import read
from sklearn.metrics import precision_recall_fscore_support as score


from scipy.stats import skew
import librosa
class ExtractFeature:

    def _calculate_delta(self,array_feature):
        '''
            Menghitung dan me return nilai delta dari feature vector matrix 
            yang diberikan. 
        '''
        rows, _ = array_feature.shape

        # membuat template array dengan jumlah row = rows dan col = 20
        deltas = np.zeros((rows,20))

        N = 2
        for i in range(rows):
            index = []
            j = 1 
            while j <= N:
                if i - j < 0:
                    first = 0
                else :
                    first = i - j 
                if i + j > rows -1 :
                    second = rows - 1
                else :
                    second = i + j 
                index.append((second,first))
                j +=1 
            deltas[i] = (array_feature[index[0][0]] - array_feature[index[0][1]] + (2 * (array_feature[index[1][0]] - array_feature[index[1][1]])))/10
        return deltas
                 

    def extract_feature_old(self,audio_path_file):
        '''
            menghasilkan 20 dimensi mfcc feature dari audion, melakukan CMS (cepstral mean subtraction)
            dan dikombinasikan dengan delta agar dapat menghasilkan 40 dimensi feature vector
        '''

        rate,audio = read(audio_path_file)
        mfcc = speech_features.mfcc(signal = audio, samplerate = rate, winlen=0.025,
                                             winstep=0.01,numcep=20,nfft=1200,appendEnergy=True)
        mfcc = mfcc.T
        # windows atau template generator
        result = np.hstack((np.mean(mfcc,axis=1)))
        return result
    def extract_feature(self,audio_path_file):
        SAMPLE_RATE = 44100
        data, _ = librosa.core.load(audio_path_file,sr = SAMPLE_RATE)
        assert _ == SAMPLE_RATE
        try:
            ft1 = librosa.feature.mfcc(data, sr = SAMPLE_RATE, n_mfcc=30)
            ft2 = librosa.feature.zero_crossing_rate(data)[0]
            ft3 = librosa.feature.spectral_rolloff(data)[0]
            ft4 = librosa.feature.spectral_centroid(data)[0]
            ft5 = librosa.feature.spectral_contrast(data)[0]
            ft6 = librosa.feature.spectral_bandwidth(data)[0]
            ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.median(ft1, axis = 1), np.min(ft1, axis = 1)))
            ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
            ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
            ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
            ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
            ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
            result = np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc))
            return result
        except:
            print('bad file')
            return np.zeros([0]*210)
ef = ExtractFeature()
