import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

frequency_sampling, audio_signal = wavfile.read("./dataset/Benjamin_Netanyau/99.wav")

audio_signal = audio_signal[:15000]

features_mfcc = mfcc(audio_signal, frequency_sampling)

print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])



features_mfcc = features_mfcc.T
print(features_mfcc)
plt.matshow(features_mfcc)
plt.title('MFCC')

filterbank_features = logfbank(audio_signal, frequency_sampling)

print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
print('Length of each feature =', filterbank_features.shape[1])

filterbank_features = filterbank_features.T
print(filterbank_features)
plt.matshow(filterbank_features)
plt.title('Filter bank')
plt.show()
