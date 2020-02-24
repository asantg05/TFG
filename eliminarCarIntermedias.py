# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 23:27:13 2020

@author: andre
"""

import librosa #Paquete de analisis de audio
import librosa.display
import IPython.display as ipd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import pathlib
import csv
import glob
import shutil


os.chdir('/Users/andre/OneDrive/Escritorio/TFG/Codigo')#C:\Users\andre\OneDrive\Escritorio\BiometryVoice\BiometriaPorVoz-master
os.getcwd()

listaIds = []
ide = "id"
numero = 10270
for i in range (0,40):
    cadena = ide+str(numero)
    listaIds.append(cadena)
    numero = numero + 1
    i = i+1
#print(listaIds)

#listaWavs = []
contador = 0
lista=[]

for carpetasIdes in listaIds:
    carpetasRaras = os.listdir(f'./datasetCompletoPruebas/{carpetasIdes}')
    for x in carpetasRaras:
        files = os.listdir(f'./datasetCompletoPruebas/{carpetasIdes}/{x}')
        for fname in files:
            fname = f'./datasetCompletoPruebas/{carpetasIdes}/{x}/{fname}'
            #print(fname)
            
            lista.append(("au") + str(contador))
            #print(lista)
            
            os.rename(fname, f'./datasetCompletoPruebas/{carpetasIdes}/{x}/{lista[contador]}.wav')
            contador = contador + 1
            #print(lista)
            #print(contador)

print("--------END---------")




    







                                        