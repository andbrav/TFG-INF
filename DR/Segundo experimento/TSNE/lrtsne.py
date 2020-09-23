from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from csv import DictWriter
from csv import writer
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

field_names = ["Dimension","Perplexity","Exaggeration","C","Accuracy"]
with open("lrtsne.csv","w", newline='') as file:
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

    print(str(dimensiones)+" "+ str(perplejidad)+" "+str(exageracion))
    
    print(y_train)
    print(X_train)
    print(list(set(y_train)))
    print(index)
    elapsed_time = perf_counter() - t0
    print("Time "+str(elapsed_time))
    for c in [0.001,0.01,0.1,1,10,100,1000]:
        t0 = perf_counter()
        print("Model with dim="+str(dimensiones)+", perplejidad="+str(perplejidad)+", exageracion="+str(exageracion)+" y C="+str(c))

        clf = LogisticRegression(C=c,max_iter=1000000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))
    
        filename = 'lr_%s_%s_%s.pckl' % (str(dimensiones),str(perplejidad),str(exageracion))
        elapsed_time = perf_counter() - t0
        print("Time "+str(elapsed_time))

        pickle.dump(clf, open(filename,'wb'), protocol=4)

        result ={"Dimension":i,"Perplexity":perplejidad,"Exaggeration":exageracion,"C":c,"Accuracy":accuracy_score(y_test, y_pred)}
        with open("lrtsne.csv","a+", newline='') as file:
                dict_writer = DictWriter(file, fieldnames=field_names)
                dict_writer.writerow(result)

    elapsed_time = perf_counter() - t0
    print("Time "+str(elapsed_time))

