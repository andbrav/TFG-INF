import os
import logging
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.datasets import mnist
from sklearn import manifold
from time import perf_counter
from openTSNE import (sklearn)


current_dir = os.getcwd()
(X_train, y_train), (X_test, y_test) = mnist.load_data(path=current_dir+'/../../../Data/MNIST/mnist.npz')
nsamples, nx, ny = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny))
nsamples, nx, ny = X_test.shape
X_test = X_test.reshape((nsamples,nx*ny))

dataset = current_dir+'/../../../Data/MNIST/mnist_mediumsmall.npz'

data = np.load(dataset)
index = data['index']


tsne_evaluations = pd.read_csv(current_dir+'/../../PrimerExperimento/Outputs/TSNE.csv')
data = tsne_evaluations[["n_components","perplexity","exaggeration","aggregate"]]


for i in [1,2,3,4,5,6,7,8,9,10,50,100,200]: 
     t0 = perf_counter()
     print("Reduciendo a "+str(i)+" dimensiones")
     data_dimensioni = data[data["n_components"] == i]
     model_data = data_dimensioni.loc[data_dimensioni['aggregate'].idxmax()]
     dimensiones = int(model_data["n_components"])
     perplejidad = int(model_data["perplexity"])
     exageracion = model_data["exaggeration"]
     tsne = sklearn.TSNE(negative_gradient_method="bh",n_components=dimensiones,perplexity=perplejidad,exaggeration=exageracion)
     X_transf = tsne.fit_transform(X_train[index,])
     elapsed_time = perf_counter() - t0
     print("Time "+str(elapsed_time))
     print("Transformando el resto")
     extra_transf = tsne.transform(X_train[[i for i in range(60000) if i not in index],])
     
     test_transf = tsne.transform(X_test)
     
     print(X_transf.shape)
     print(extra_transf.shape)
     X_transf = np.append( X_transf,extra_transf,axis=0)
     print(X_transf.shape)

     index2 = np.append(index,np.array([i for i in range(60000) if i not in index]))

     np.savez_compressed('tsnetrain%s_%s_%s.npy' % (str(dimensiones),str(perplejidad),str(exageracion)), X=X_transf, Y =y_train[index2], index=index2)
     np.savez_compressed('tsnetest%s_%s_%s.npy' % (str(dimensiones),str(perplejidad),str(exageracion)), X=test_transf, Y =y_test)
     elapsed_time = perf_counter() - t0
     print("Time "+str(elapsed_time))
     

