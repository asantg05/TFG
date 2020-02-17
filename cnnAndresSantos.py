# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:25:43 2020

@author: andre
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import utils

K.clear_session()

#Conteo de imagenes en la carpeta espectrogramas
imgpath = 'E:/TrabajoFinGrado/Data2Reducido'

images = []#Aqui añadimos imagenes
directories = []#Directorios
dircount = []#Cuenta de imagenes en directorios
prevRoot=''
cant=0

print("leyendo imagenes de ",imgpath)

for root, dirnames, filenames in os.walk(imgpath):#Se recorre TODO de la carpeta espectrogramas
    for filename in filenames:#Establecemos regla para los archivos
        if re.search("\.png", filename):#El paquete "re" es utilizado para busqueda de strings, por ello ponemos regla: aquellos que sean .png
            cant=cant+1#Con este contador, contamos imagenes de cada directorio
            filepath = os.path.join(root, filename)#Unimos directorio actual con el filename
            image = plt.imread(filepath)#Se lee la imagen del file para luego ser introducida en un array
            images.append(image)#Añadimos image a el array
            b = "Leyendo..." + str(cant)
            print (b, end="\r")#Se imprime el numero de imagenes
            if prevRoot !=root:
                print(root, cant)#Se imprime ruta del directorio id0001, id0002...
                prevRoot=root
                directories.append(root)#lista de directorios encontrados
                dircount.append(cant)#añadimos numero de imagenes encontradas en cada directorio al array dircount
                cant=0#Se actualiza a 0 la cuenta de imagenes
dircount.append(cant)

dircount = dircount[1:]
dircount[0]=dircount[0]+1
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))

labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)#Se añaden las etiquetas de las imagenes
    indice=indice+1
print("Cantidad etiquetas creadas: ",len(labels))

#Guardando el array de imagenes como npy.file
np.save('images.npy', images)

#Representacion OneHot de las labels
y_labels_one_hot = to_categorical(labels)

#Guardando las etiquetasOneHot como npy.file
np.save('y_labels_one_hot.npy', y_labels_one_hot)

from sklearn.utils import shuffle

#Barajar y dividir el conjunto de datos en el train y el de validacion
filenames_shuffled, y_labels_one_hot_shuffled = shuffle(images, y_labels_one_hot)

#Guardamos los conjuntos barajados
#Despues se puede cargar con np.load()
np.save('y_labels_one_hot_shuffled.npy', y_labels_one_hot_shuffled)
np.save('filenames_shuffled.npy', filenames_shuffled)

# Convertimos conjunto de imagenes en array de tipo numpy
filenames_shuffled_numpy = np.array(filenames_shuffled)

#Separamos en train y validacion
X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(#X=imagenes; y=labels
    filenames_shuffled_numpy, y_labels_one_hot_shuffled, test_size=0.2, random_state=1)

print(X_train_filenames.shape) # (3800,)
print(y_train.shape)           # (3800, 12)

print(X_val_filenames.shape)   # (950,)
print(y_val.shape)             # (950, 12)

# Guardamos estos files de train y validacion de nuevo como .npy
np.save('X_train_filenames.npy', X_train_filenames)
np.save('y_train.npy', y_train)

np.save('X_val_filenames.npy', X_val_filenames)
np.save('y_val.npy', y_val)

#Creamos un CustomGeneratos que cargará nuestro dataset en lotes
class My_Custom_Generator(keras.utils.Sequence):
  
  def __init__(self, image_filenames, labels, batch_size):
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self):
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx):
    batch_x = self.image_filenames[idx * self.batch_size: (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([
            resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
               for file_name in batch_x])/255.0, np.array(batch_y)
    
#Creamos instancias de BatchGenerator
batch_size = 32

my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size)
my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size)

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.transform import resize

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical

model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu',input_shape=(80,80,3)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = (5,5), activation ='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters = 128, kernel_size = (5,5), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 256, kernel_size = (5,5), activation ='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters = 256, kernel_size = (5,5), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu")) #Fully connected layer
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(60, activation = "relu")) #Fully connected layer
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(12, activation = "softmax")) #Classification layer or output layer

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(generator=my_training_batch_generator,
                   steps_per_epoch = int(3800 // batch_size),
                   epochs = 10,
                   verbose = 1,
                   validation_data = my_validation_batch_generator,
                   validation_steps = int(950 // batch_size))


