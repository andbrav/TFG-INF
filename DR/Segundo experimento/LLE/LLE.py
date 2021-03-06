import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.datasets import mnist
from sklearn import manifold
from time import perf_counter

current_dir = os.getcwd()
(X_train, y_train), (X_test, y_test) = mnist.load_data(path=current_dir+'/../../../Data/MNIST/mnist.npz')
nsamples, nx, ny = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny))
nsamples, nx, ny = X_test.shape
X_test = X_test.reshape((nsamples,nx*ny))

dataset = "current_dir+'/../../../Data/MNIST/mnist_medium.npz"

data = np.load(dataset)
index = data['index']


lle_evaluations = pd.read_csv(current_dir+'/../../PrimerExperimento/Outputs/LLE.csv')
data = lle_evaluations [["n_components","n_neighbors","eigen_solver","n_jobs","aggregate"]]


for i in [1,2,3,4,5,6,7,8,9,10,50,100,200]:
     t0 = perf_counter()
     print("Reduciendo a "+str(i)+" dimensiones")
     data_dimensioni = data[data["n_components"] == i]
     model_data = data_dimensioni.loc[data_dimensioni['aggregate'].idxmax()]
     dimensiones = model_data["n_components"]
     vecinos = model_data["n_neighbors"]
     solver = model_data["eigen_solver"]
     jobs = model_data["n_jobs"]
     lle = manifold.LocallyLinearEmbedding(n_components=dimensiones, n_neighbors=vecinos, eigen_solver=solver, n_jobs=jobs)
     X_transf = lle.fit_transform(X_train[index,])
     print("Transformando el resto")
     extra_transf = lle.transform(X_train[[i for i in range(60000) if i not in index]
,])
     test_transf = lle.transform(X_test)
     print(X_transf.shape)
     print(extra_transf.shape)
     X_transf = np.append( X_transf,extra_transf,axis=0)
     print(X_transf.shape)
     pickle.dump(lle, open("lle%s_%s_%s.pckl" % (str(dimensiones),str(vecinos),str(solver)),'wb'))
     np.save('lletrain%s_%s_%s.npy' % (str(dimensiones),str(vecinos),str(solver)), X_transf)
     np.save('lletest%s_%s_%s.npy' % (str(dimensiones),str(vecinos),str(solver)), test_transf)
     elapsed_time = perf_counter() - t0
     print("Time "+str(elapsed_time))
     

