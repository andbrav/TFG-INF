from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from csv import DictWriter
from csv import writer
import os
from tensorflow.keras.datasets import mnist
import logging
import pickle
import numpy as np
import umap
import pandas as pd
from time import perf_counter
from openTSNE import (sklearn)
import math

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

def onepixeltranslations(image):
	(x,y) = image.shape
	row = np.zeros((1,x))
	column = np.zeros((y,1)) 
	up = np.append(image,row,axis=0)[1:,:]
	down = np.append(row,image,axis=0)[:-1,:]
	left = np.append(image,column,axis=1)[:,1:]
	right = np.append(column,image,axis=1)[:,:-1]
	up_left = np.append(up,column,axis=1)[:,1:]
	up_right = np.append(column,up,axis=1)[:,:-1]
	down_left = np.append(down,column,axis=1)[:,1:]
	down_right = np.append(column,down,axis=1)[:,:-1]
	return (up,down,left,right,up_left,up_right,down_left,down_right)


current_dir = os.getcwd()
(X_orig, y_orig), (X_torig,y_test) = mnist.load_data(path=current_dir+'/../../../Data/MNIST/mnist.npz')

nsamples, nx, ny = X_orig.shape
X_orig2 = X_orig.reshape((nsamples,nx*ny))
nsamples, nx, ny = X_torig.shape
X_torig2 = X_torig.reshape((nsamples,nx*ny))

tsne_evaluations = pd.read_csv(current_dir+'/../../PrimerExperimento/Outputs/TSNE.csv')
data = tsne_evaluations[["n_components","perplexity","exaggeration","aggregate"]]

dataset = current_dir+"/../../../Data/MNIST/mnist_mediumsmall.npz"
data2 = np.load(dataset)
index = data2['index']


field_names = ["Dimension","Accuracy_standard","Accuracy_vsvm"]
with open("svmtsne.csv","w", newline='') as file:
    filewriter = writer(file, delimiter=',')
    filewriter.writerow(field_names)

for i in [1,2,3,4,5,6,7,8,9]:
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

    print(str(dimensiones)+" "+ str(perplejidad)+" "+str(exageracion))
    
    print(y_train)
    print(X_train)
    print(list(set(y_train)))
    print(index)
    
    print("Model with dim="+str(dimensiones)+" y perplejidad="+str(perplejidad))
    svclassifier = SVC()
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ac_st = accuracy_score(y_test, y_pred)

    # Virtual data creation
    print("Number of support points "+str(svclassifier.support_.shape[0]))
    print(X_orig[index2[svclassifier.support_]].shape)

    VX_train = np.array([image for SV in X_orig[index2[svclassifier.support_]] for image in onepixeltranslations(SV)])
    Vy_train = np.array([label for label in  y_orig[index2[svclassifier.support_]] for i in range(8)])
    print(VX_train.shape)
    print(Vy_train.shape)

    nsamples, nx, ny = VX_train.shape
    VX_train = VX_train.reshape((nsamples,nx*ny))
    VX_train = tsne.transform(VX_train)

    VX_train = np.append(VX_train,X_train[svclassifier.support_],axis=0)
    Vy_train = np.append(Vy_train,y_train[svclassifier.support_])

    print(VX_train.shape)
    print(Vy_train.shape)
    svclassifier2 = SVC()
    svclassifier2.fit(VX_train, Vy_train)
    y_pred = svclassifier2.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ac_svm = accuracy_score(y_test, y_pred)

    print("Number of support points "+str(svclassifier2.support_.shape[0]))
    filename = 'svm_base_%s_%s_%s.pckl' % (str(dimensiones),str(perplejidad),str(exageracion))
    elapsed_time = perf_counter() - t0
    print("Time "+str(elapsed_time))

    pickle.dump(svclassifier2, open(filename,'wb'), protocol=4)
    result ={"Dimension":dim,"Accuracy_standard":ac_st,"Accuracy_vsvm":ac_svm}
    with open("svmtsne.csv","a+", newline='') as file:
            dict_writer = DictWriter(file, fieldnames=field_names)
            dict_writer.writerow(result)







