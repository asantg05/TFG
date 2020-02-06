# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:29:38 2020

@author: andre
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:15:29 2020

@author: andre

"""

import librosa #Paquete de analisis de audio
import librosa.display
import matplotlib.pyplot as plt

songname = '/Users/andre/OneDrive/Escritorio/00003.wav'

#cmap = plt.get_cmap('magma')
plt.figure(figsize=(15,15))

#Mfccs
y, sr = librosa.load(songname, mono=True, duration=8)
#librosa.feature.mfcc(y=y, sr=sr) # ,hop_length=1024, htk=True)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
librosa.feature.mfcc(S=librosa.power_to_db(S))
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

librosa.display.specshow(mfccs)
plt.colorbar()
plt.tight_layout()
plt.savefig('/Users/andre/OneDrive/Escritorio/mfcc.png') #Guardamos en Data/Idxxxxx, cambiamos el punto por nada
plt.show()













