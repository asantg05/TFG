# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:31:55 2020

@author: andre
"""

from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

#tendremos 32 filtros usando una ventana de 5×5 para la capa convolucional y una ventana de 2×2 para la capa de pooling.
model = models.Sequential()

model.add(layers.Conv2D(32,(5,5),activation='relu',input_shape=(28,28,1)))#función de activación ReLU, tensor de entrada de tamaño (28, 28, 1)
model.add(layers.MaxPooling2D((2, 2)))#ventana de 2×2 para la capa de pooling

"""El número de parámetros de la capa conv2D corresponde a la matriz de pesos W de 5×5
 y un sesgo b para cada uno de los filtros es 832 parámetros (32 × (25+1))"""

#crearemos un segundo grupo de capas que tendrá 64 filtros con una ventana de 5×5 en la capa convolucional y una de 2×2 en la capa de pooling
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

"""El siguiente paso, ahora que tenemos 64 filtros de 4×4, consiste en añadir una capa densamente conectada (densely connected layer),
 que servirá para alimentar una capa final de softmax"""
"""se tiene primero que aplanar el tensor de 3D a uno de 1D. Nuestra salida (4,4,64) se debe aplanar a un vector de (1024) antes de aplicar el Softmax."""
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

model.summary()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(train_images, train_labels,
          batch_size=100,
          epochs=5,
          verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)