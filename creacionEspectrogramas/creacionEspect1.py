# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:21:55 2020

@author: andre
"""

from scipy.io import wavfile # scipy library to read wav files
import numpy as np
import matplotlib.pyplot as plt


AudioName = "/Users/andre/OneDrive/Escritorio/00004.wav" # Audio File
fs, Audiodata = wavfile.read(AudioName)

from scipy import signal
N = 512 #Number of point in the fft
f, t, Sxx = signal.spectrogram(Audiodata, fs,window = signal.blackman(N),nfft=N)
plt.figure()
plt.pcolormesh(t, f,10*np.log10(Sxx)) # dB spectrogram
#plt.pcolormesh(t, f,Sxx) # Lineal spectrogram
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [seg]')
#plt.title('Spectrogram with scipy.signal',size=16);
plt.savefig('/Users/andre/OneDrive/Escritorio/espec.png') #Guardamos la imagen

plt.show()

