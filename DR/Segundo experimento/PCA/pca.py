import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.datasets import mnist
from sklearn import (decomposition,manifold)
from time import perf_counter

current_dir = os.getcwd()
(X_train, y_train), (X_test, y_test) = mnist.load_data(path=current_dir+'/../../../Data/MNIST/mnist.npz')
nsamples, nx, ny = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny))
nsamples, nx, ny = X_test.shape
X_test = X_test.reshape((nsamples,nx*ny))

dataset = current_dir+"/../../../Data/MNIST/mnist_medium.npz"

data = np.load(dataset)
index = data['index']

pca_evaluations = pd.read_csv(current_dir+'/../../PrimerExperimento/Outputs/PCA.csv')
data = pca_evaluations[["n_components","aggregate"]]


for i in [1,2,3,4,5,6,7,8,9,10,50,100,200]:
     t0 = perf_counter()
     print("Reduciendo a "+str(i)+" dimensiones")
     data_dimensioni = data[data["n_components"] == i]
     model_data = data_dimensioni.loc[data_dimensioni['aggregate'].idxmax()]
     dimensiones = int(model_data["n_components"])

     pca = decomposition.PCA(n_components=dimensiones)

     X_transf = pca.fit_transform(X_train[index,])
     print("Transformando el resto")
     extra_transf = pca.transform(X_train[[i for i in range(60000) if i not in index]
,])
     
     test_transf = pca.transform(X_test)
     
     print(X_transf.shape)
     print(extra_transf.shape)
     X_transf = np.append( X_transf,extra_transf,axis=0)
     print(X_transf.shape)

     index2 = np.append(index,np.array([i for i in range(60000) if i not in index]))

     pickle.dump(pca, open("pca%s.pckl" % (str(dimensiones)),'wb'),protocol=4)
     np.savez_compressed('pcatrain%s.npy' % (str(dimensiones)), X=X_transf, Y =y_train[index2], index=index2)
     np.savez_compressed('pcatest%s.npy' % (str(dimensiones)), X=test_transf, Y =y_test)
     elapsed_time = perf_counter() - t0
     print("Time "+str(elapsed_time))
     

