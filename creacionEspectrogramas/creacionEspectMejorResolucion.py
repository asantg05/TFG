# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:15:29 2020

@author: andre

cmaps['Perceptually Uniform Sequential'] = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']

cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

cmaps['Sequential (2)'] = [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']
"""

import librosa #Paquete de analisis de audio
import librosa.display
import matplotlib.pyplot as plt

songname = '/Users/andre/OneDrive/Escritorio/00003.wav'

#cmap = plt.get_cmap('magma')
cmap = plt.get_cmap('inferno')
plt.figure(figsize=(10,10))

#Espectrograma 692 con 2040

y, sr = librosa.load(songname, mono=True, duration=8)#Lectura de Espectrogramas
plt.specgram(y, NFFT=256, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')#Creacion de Espectrogramas
plt.axis('off')
plt.savefig('/Users/andre/OneDrive/Escritorio/especMalo.png') #Guardamos en Data/Idxxxxx, cambiamos el punto por nada
plt.show()
plt.clf()#plt.clf() will just clear the figure. You can still paint another plot onto it. --> https://stackoverflow.com/questions/16661790/difference-between-plt-close-and-plt-clf















