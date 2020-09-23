import os
import logging
import numpy as np
import pandas as pd
import pickle
import umap
from tensorflow.keras.datasets import mnist
from sklearn import manifold
from time import perf_counter

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

current_dir = os.getcwd()
(X_train, y_train), (X_test, y_test) = mnist.load_data(path=current_dir+'/../../../Data/MNIST/mnist.npz')
nsamples, nx, ny = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny))
nsamples, nx, ny = X_test.shape
X_test = X_test.reshape((nsamples,nx*ny))

dataset = current_dir+"/../../../Data/MNIST/mnist_mediumsmall.npz"

data = np.load(dataset)
index = data['index']


umap_evaluations = pd.read_csv(current_dir+'/../../PrimerExperimento/Outputs/UMAP.csv')
data = umap_evaluations[["n_components","n_neighbors","min_dist","aggregate"]]


for i in [1,2,3,4,5,6,7,8,9,10,100,50,200]: 
     t0 = perf_counter()
     print("Reduciendo a "+str(i)+" dimensiones")
     data_dimensioni = data[data["n_components"] == i]
     model_data = data_dimensioni.loc[data_dimensioni['aggregate'].idxmax()]
     dimensiones = model_data["n_components"]
     vecinos = int(model_data["n_neighbors"])
     min_dist = model_data["min_dist"]
     ump = umap.UMAP(n_components=dimensiones,n_neighbors=vecinos,min_dist=min_dist)
     X_transf = ump.fit_transform(X_train[index,])
     elapsed_time = perf_counter() - t0
     print("Time "+str(elapsed_time))
     print("Transformando el resto")
     extra_transf = ump.transform(X_train[[i for i in range(60000) if i not in index],])
     
     test_transf = ump.transform(X_test)
     
     print(X_transf.shape)
     print(extra_transf.shape)
     X_transf = np.append( X_transf,extra_transf,axis=0)
     print(X_transf.shape)

     index2 = np.append(index,np.array([i for i in range(60000) if i not in index]))

     pickle.dump(ump, open("umap%s_%s_%s.pckl" % (str(dimensiones),str(vecinos),str(min_dist)),'wb'),protocol=4)
     np.savez_compressed('umaptrain%s_%s_%s.npy' % (str(dimensiones),str(vecinos),str(min_dist)), X=X_transf, Y =y_train[index2], index=index2)
     np.savez_compressed('umaptest%s_%s_%s.npy' % (str(dimensiones),str(vecinos),str(min_dist)), X=test_transf, Y =y_test)
     elapsed_time = perf_counter() - t0
     print("Time "+str(elapsed_time))
     

