# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 10:51:00 2024

@author: balav
"""

# Artificial Neural Network

# Importing the libraries
import numpy as np #numerical operations
import pandas as pd #data cleaning , data analysis -> dataframe,Series
import tensorflow as tf




# Importing the dataset
dataset = pd.read_csv("C:\\Users\\balav\\OneDrive\\Desktop\\Imagecon\\Deep Learning Dataset\\Churn_Modelling.csv")
dataset.head(60)
dataset.columns
dataset.describe()
dataset.dtypes
dataset.info()

dataset.isna().any()
#isna() -> NaN(not an number), None


dataset["Exited"].value_counts()











X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


# Encoding categorical data
# Label Encoding 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

X


# One Hot Encoding 

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)






# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                    test_size = 0.2, random_state = 0)









# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)






#Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))




#Training the ANN

# Compiling the ANN #categorical_crossentropy----multiclass
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
            metrics = ['acc'])









# Training the ANN on the Training set
model_history=ann.fit(X_train, y_train, batch_size = 8000, 
                      epochs = 500,validation_split = 0.1)

#10% of the training data will be used as validation data 









# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5) #1 , 0









# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)








#model performance
ann.evaluate(X_test, y_test)


#visualization
#training accuracy and validation accuracy over epochs
import matplotlib.pyplot as plt

#accuracy
plt.figure(dpi=300)
plt.plot(model_history.history['acc']) #Plots the training accuracy at each epoch

plt.plot(model_history.history['val_acc']) #Plots the validation accuracy at each epoch 

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Training', 'Validation'], loc='upper right')

#loss
#plotting the training loss and validation loss
plt.figure(dpi=300)
plt.plot(model_history.history['loss']) #Plots the training loss at each epoch 

plt.plot(model_history.history['val_loss']) #Plots the validation loss at each epoch

plt.xlabel('Epochs')

plt.ylabel('loss')

plt.legend(['Training', 'Validation'], loc='upper right')