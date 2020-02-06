# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:17:51 2020

@author: andre
"""

import keras
import keras_resnet.models

shape, classes = (32, 32, 3), 10

x = keras.layers.Input(shape)

model = keras_resnet.models.ResNet50(x, classes=classes)

model.compile("adam", "categorical_crossentropy", ["accuracy"])

(training_x, training_y), (_, _) = keras.datasets.cifar10.load_data()

training_y = keras.utils.np_utils.to_categorical(training_y)

model.fit(training_x, training_y)