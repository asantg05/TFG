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
from os import rmdir


os.chdir('/Users/andre/OneDrive/Escritorio/TFG/Codigo')#C:\Users\andre\OneDrive\Escritorio\BiometryVoice\BiometriaPorVoz-master
os.getcwd()

listaIds = [] #En vez de usar el csv, metemos en una lista los ides
ide = "id"
numero = 10270
for i in range (0,40):
    cadena = ide+str(numero)
    listaIds.append(cadena)
    numero = numero + 1
    i = i+1
#print(listaIds)

contador = 0
lista=[]#Esta es la lista auxiliar para cambiar el nombre de los wavs

for carpetasIdes in listaIds:
    carpetasRaras = os.listdir(f'./datasetCompletoPruebas/{carpetasIdes}')
    for x in carpetasRaras:
        #rmdir(f'./datasetCompletoPruebas/{carpetasIdes}/{x}') #NO ME DEJA ELIMINAR CARPETAS RARAS POR PERMISOS
        
        files = os.listdir(f'./datasetCompletoPruebas/{carpetasIdes}/{x}')
        for fname in files:
            fname = f'./datasetCompletoPruebas/{carpetasIdes}/{x}/{fname}'
            #print(fname)
            
            lista.append(("au") + str(contador))
            #print(lista)
            
            #ASI RENOMBRAMOS - HACEMOS ESTO LO PRIMERO
            #os.rename(fname, f'./datasetCompletoPruebas/{carpetasIdes}/{x}/{lista[contador]}.wav')
            
            #ASI MOVEMOS - LO SEGUNDO
            #shutil.move(fname, f'./datasetCompletoPruebas/{carpetasIdes}/{lista[contador]}.wav')
            
            contador = contador + 1
            #print(contador) #Numero de wavs que hay en todo el dataset

print("--------END---------")




    







                                        