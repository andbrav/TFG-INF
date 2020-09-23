from sklearn.metrics import classification_report, confusion_matrix
import os
from tensorflow.keras.datasets import mnist
import pickle
import numpy as np
import pandas as pd
from time import perf_counter
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from csv import DictWriter
from csv import writer
from openTSNE import (sklearn)
import math

current_dir = os.getcwd()
(X_orig, y_orig), (X_torig,y_test) = mnist.load_data(path=current_dir+'/../../../Data/MNIST/mnist.npz')

nsamples, nx, ny = X_orig.shape
X_orig2 = X_orig.reshape((nsamples,nx*ny))
nsamples, nx, ny = X_torig.shape
X_torig2 = X_torig.reshape((nsamples,nx*ny))
y_test = to_categorical(y_test)

tsne_evaluations = pd.read_csv(current_dir+'/../../PrimerExperimento/Outputs/TSNE.csv')
data = tsne_evaluations[["n_components","perplexity","exaggeration","aggregate"]]

dataset = current_dir+"/../../../Data/MNIST/mnist_mediumsmall.npz"
data2 = np.load(dataset)
index = data2['index']

field_names = ["Dimension","Perplexity","Exaggeration","Accuracy"]
with open("mptsne.csv","w", newline='') as file:
    filewriter = writer(file, delimiter=',')
    filewriter.writerow(field_names)

for i in [1,2,3,4,5,6,7,8,9,10]:
    t0 = perf_counter()
    print("Reduciendo a "+str(i)+" dimensiones")
    data_dimensioni = data[data["n_components"] == i]
    model_data = data_dimensioni.loc[data_dimensioni['aggregate'].idxmax()]
    dimensiones = int(model_data["n_components"])
    perplejidad = int(model_data["perplexity"])
    exageracion = model_data["exaggeration"]
    if math.isnan(exageracion):
        exageracion = None
    tsne = sklearn.TSNE(negative_gradient_method="bh",n_components=dimensiones,perplexity=perplejidad,exaggeration=exageracion)
    X_transf = tsne.fit_transform(X_orig2[index,])
    extra_transf = tsne.transform(X_orig2[[i for i in range(60000) if i not in index],])
    X_train = np.append( X_transf,extra_transf,axis=0)
    X_test = tsne.transform(X_torig2)
    index2 = np.append(index,np.array([i for i in range(60000) if i not in index]))
    y_train = y_orig[index2]
    y_train = to_categorical(y_train)
    print(str(dimensiones)+" "+ str(perplejidad)+" "+str(exageracion))
    
    print(y_train)
    print(X_train)
    print(index)
    elapsed_time = perf_counter() - t0
    print("Time "+str(elapsed_time))
    t0 = perf_counter()
    print("Model with dim="+str(dimensiones)+", perplejidad="+str(perplejidad)+", exageracion="+str(exageracion))
    verbose, epochs, batch_size = 0, 10, 32
    n_outputs = 10
    model = Sequential()
    model.add(Dense(100, activation='relu',input_dim=dimensiones))
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

    result ={"Dimension":i,"Perplexity":perplejidad,"Exaggeration":exageracion,"Accuracy":accuracy}
    with open("mptsne.csv","a+", newline='') as file:
        dict_writer = DictWriter(file, fieldnames=field_names)
        dict_writer.writerow(result)



