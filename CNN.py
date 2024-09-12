# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:46:52 2024

@author: balav
"""

# Importing the libraries
import numpy as np #numerical operations
import pandas as pd #data cleaning , data analysis -> dataframe,Series
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

training_set = training_datagen.flow_from_directory("C:\\Users\\balav\\OneDrive\\Desktop\\Imagecon\\Deep Learning Dataset\\CNN\\training_set"
                                                    ,target_size=(64,64),batch_size=32,class_mode='binary')

test_datagen= ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_set= training_datagen.flow_from_directory("C:\\Users\\balav\\OneDrive\\Desktop\\Imagecon\\Deep Learning Dataset\\CNN\\test_set"
                                                    ,target_size=(64,64),batch_size=32,class_mode='binary')

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

cnn.add(tf.keras.layers.Dense(units=1,activation='relu'))

cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

cnn.fit(training_set,validation_data= test_set,epochs=25)

from keras.preprocessing import image
test_image= tf.keras.utils.load_img("C:\\Users\\balav\\OneDrive\\Desktop\\Imagecon\\Deep Learning Dataset\\CNN\\single_prediction\\cat_or_dog_1.jpg")
test_image= tf.keras.utils.img_to_array(test_image)
test_image= np.expand_dims(test_image,axis=0)
result= cnn.predict(test_image)
if result[0][0] ==1 :
    prediction='dog'
else:
    prediction='cat'

print(prediction)

history = cnn.fit(training_set,validation_data= test_set,epochs=25)

test_loss, test_acc= cnn.evaluate(test_set)

print("Test loss:",test_loss)
print("Test accuracy:",test_acc)  

plt.figure(dpi=300)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy') 
plt.xlable('Epoch')
plt.legend(['train','val'],loc='best')
plt.show()                               
                                    