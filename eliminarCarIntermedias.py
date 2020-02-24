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

"""directorio = '/Users/andre/OneDrive/Escritorio/TFG/Codigo/DatasetCompletoPruebas'

for root, subdirList, fileList in os.walk(directorio):#dirName el nombre de la carpeta
    print(subdirList)#subdirList es una tupla donde se encuentran todas las carpetas del directorio
    print('Directorio encontrado: %s' % root)
    for fname in fileList:
        print('\t%s' % fname)"""

listaIds = []
ide = "id"
numero = 10270
for i in range (0,40):
    cadena = ide+str(numero)
    listaIds.append(cadena)
    numero = numero + 1
    i = i+1
print(listaIds)

listaWavs = []
contador = 1
listaNumeros = []
otroContador=1

for carpetasIdes in listaIds:
    carpetasRaras = os.listdir(f'./datasetCompletoPruebas/{carpetasIdes}')
    for x in carpetasRaras:
        files = os.listdir(f'./datasetCompletoPruebas/{carpetasIdes}/{x}')
        for fname in files:
            fname = f'./datasetCompletoPruebas/{carpetasIdes}/{x}/{fname}'
            
            cabeza_cola = os.path.split(fname)#separamos para que quede: 00001.wav
            nombreArchivoWav = cabeza_cola[1]#estabecemos 00001.wav
            a = os.path.splitext(nombreArchivoWav)#separamos para que quede 00001
            numId = a[0]#00001
            #print(numId)
            
            listaNumeros.append(numId)
            #print(listaNumeros)
            
            for z in listaNumeros:
                z = z.replace("0","")
                z = "audio" + z
                listaNumeros.append(z)
                
            print(listaNumeros)

print("--------END---------")
            
            
"""if fname == firstFile:#Si 00001 nuevo coincide con el 00001 antiguo...
                fname.replace(fname[4], f"{contador}")#sustituimos el 0000-1-.wav por el contador -> 00028.wav
                if contador>=100:
                    fname = fname[2:]#Recorte de 2 ceros
                elif contador>=10:
                    fname = fname[1:]#Recorte del cero del principio

            else:#Este bucle filtra los que no son los primeros de las listas 00001.wav
        
                if contador<100:
                    if fname[3].find("0"):#Si en esos 2 numeros, hay un cero... 000[0]2->00029
                        fname.replace(fname[4], f"{contador}")
                        fname = fname[1:]#Recorte del cero del principio
                    else:#Aqui no hace falta borrar ceros
                        fname.replace(fname[4:5], f"{contador}")#000[10] por 000[45]
        
                elif contador>=100:
                    if fname[3].find("0"):
                        fname.replace(fname[4], f"{contador}")
                        fname = fname[2:]#Recorte de 2 ceros
                    else:
                        fname.replace(fname[4], f"{contador}")
                        fname = fname[1:]#Recorte del cero del principio"""
            #print(fname)
            #shutil.move( fname , f"./datasetCompletoPruebas/{carpetasIdes}")            
            #contador=contador+1
            
        #print(files)
    #print(carpetasRaras)


#dirs = os.listdir('./datasetCompletoPruebas/id10270')
#print (dirs)
#contador = 1
    
"""if fname == firstFile:#Si 00001 nuevo coincide con el 00001 antiguo...
    fname.replace(fname[4], f"{contador}")#sustituimos el 0000-1-.wav por el contador -> 00028.wav
    if contador>=100:
        fname = fname[2:]#Recorte de 2 ceros
    elif contador>=10:
        fname = fname[1:]#Recorte del cero del principio

else:#Este bucle filtra los que no son los primeros de las listas 00001.wav

    if contador<100:
        if fname[3].find("0"):#Si en esos 2 numeros, hay un cero... 000[0]2->00029
            fname.replace(fname[4], f"{contador}")
            fname = fname[1:]#Recorte del cero del principio
        else:#Aqui no hace falta borrar ceros
            fname.replace(fname[4:5], f"{contador}")#000[10] por 000[45]

    elif contador>=100:
        if fname[3].find("0"):
            fname = fname[2:]#Recorte de 2 ceros
        else:
            fname = fname[1:]#Recorte del cero del principio"""
            

    

"""datasetCsv=[]
with open('/Users/andre/OneDrive/Escritorio/TFG/Codigo/People3.csv') as csvDataFile:
    datasetCsv = csv.reader(csvDataFile)
    for id in datasetCsv:
        for dirName, subdirList, fileList in os.walk(f'./datasetCompletoPruebas/{id}'):#dirName el nombre de la carpeta
            print(subdirList)#subdirList es una lista donde se encuentran todas las carpetas del directorio
            #print('Directorio encontrado: %s' % dirName)
            #for sub in subdirList:#Subdirectorios tipo: "5r0dWxy17C8"
                for fname in fileList:
                    fileName = f'./datasetCompletoPruebas/{id}/{dr}/{fname}.wav'#Este siempre es el primer file
                    #cabeza_cola=os.path.split(fname)
                    #nombre = cabeza_cola[1]
                    print(fileName)
                    #shutil.move( fname , "F:/TrabajoFinGrado/datasetCompletoPruebas/{}".format(id))"""

    







                                        