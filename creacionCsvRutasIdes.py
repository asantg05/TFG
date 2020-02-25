# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 01:47:51 2020

@author: andre
"""

import csv
import pandas as pd
import os
import pathlib

os.chdir('F:/TrabajoFinGrado')
os.getcwd()

cabeceras = [["ruta","id"]]

datasetCsv=[]
with open('./People3.csv') as csvDataFile:
    datasetCsv = csv.reader(csvDataFile)
    
    myFile = open('./rutas_ides.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(cabeceras)

#Filtramos por files de cada id para obtener su ruta completa
        for a in datasetCsv:#Recorremos el CSV, siendo a cada id
            for root, dirs, files in os.walk(f'F:/TrabajoFinGrado/DataNuevo_720x720/{a}'):#Recorremos carpeta donde estan todos los png
                for file in files:#Filtramos solo por Files
                    if file.endswith(".png"):#Filtramos solo aquellos que tengan .wav
                        songname = f'F:/TrabajoFinGrado/DataNuevo_720x720/{a}/{file}'#'F:/TrabajoFinGrado/Data3_500x500/id10270/au0.png'
                        print('%s' % songname)
                        x = [songname,a]
                        
                        writer.writerow(x)
    
    myFile.close()
csvDataFile.close()
                    





    
    
    
    
    
    
    
    
    
