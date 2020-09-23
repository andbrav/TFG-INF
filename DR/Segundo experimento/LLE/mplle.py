from sklearn.metrics import classification_report, confusion_matrix
import os
from tensorflow.keras.datasets import mnist
import pickle
import numpy as np
from time import perf_counter
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from csv import DictWriter
from csv import writer
import itertools

current_dir = os.getcwd()
(X_orig, y_orig), (_,_) = mnist.load_data(path=current_dir+'/../../../Data/MNIST/mnist.npz')


#Parameters
dr = [1,2,3,4,5,6,7,8,9,10,50,100,200]
v = [6,5,6,5,6,4,6,4,6,9,9,9,7]

dataset = current_dir+"/../../../Data/MNIST/mnist_medium.npz"
data = np.load(dataset)
index = data['index']
index = np.append(index,np.array([i for i in range(60000) if i not in index]))

field_names = ["Dimension","Vecinos","Accuracy"]
with open("mplle.csv","w", newline='') as file:
    filewriter = writer(file, delimiter=',')
    filewriter.writerow(field_names)



for dim in dr:
    t0 = perf_counter()
    vecinos = v[dr.index(dim)]
    print(str(dim)+" "+ str(vecinos))
    train = np.load('lletrain%s_%s_dense.npz'  % (str(dim),str(vecinos)))
    test = np.load('lletest%s_%s_dense.npz'  % (str(dim),str(vecinos)))
    X_train = train['X']
    y_train = train['Y']
    index =  train['index']
    X_test = test['X']
    y_test = test['Y'] 
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(X_train.shape)
    print(y_train.shape)
    print(index)
    
    print("Model with dim="+str(dim)+", vecinos="+str(vecinos))
    verbose, epochs, batch_size = 0, 10, 32
    n_outputs = 10
    model = Sequential()
    model.add(Dense(100, activation='relu',input_dim=dim))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print(accuracy)
    elapsed_time = perf_counter() - t0
    print("Time "+str(elapsed_time)) 

    result ={"Dimension":dim,"Vecinos":vecinos,"Accuracy":accuracy}
    with open("mplle.csv","a+", newline='') as file:
            dict_writer = DictWriter(file, fieldnames=field_names)
            dict_writer.writerow(result)

