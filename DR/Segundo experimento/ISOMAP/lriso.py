from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
from csv import DictWriter
from csv import writer
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
from time import perf_counter
from sklearn import preprocessing
import itertools

current_dir = os.getcwd()
(X_orig, y_orig), (_,_) = mnist.load_data(path=current_dir+'/../../../Data/MNIST/mnist.npz')


#Parameters
dr = [1,2,3,4,5,6,7,8,9,10,50,100,200]
C  = [0.001,0.01,0.1,1,10,100,1000]
param = list(itertools.product(dr, C))
v = [9,4,3,8,8,9,9,9,9,9,9,9,9]
dataset = current_dir+'/../../../Data/MNIST/mnist_medium.npz'
data = np.load(dataset)
index = data['index']
index = np.append(index,np.array([i for i in range(60000) if i not in index]))

field_names = ["Dimension","Vecinos","C","Accuracy"]
with open("lriso.csv","w", newline='') as file:
    filewriter = writer(file, delimiter=',')
    filewriter.writerow(field_names)



for d in param:
    t0 = perf_counter()
    dim = d[0]
    c = d[1]
    vecinos = v[dr.index(dim)]
    print(str(dim)+" "+ str(vecinos))
    train = np.load('isotrain%s_%s_dense.npz'  % (str(dim),str(vecinos)))
    test = np.load('isotest%s_%s_dense.npz'  % (str(dim),str(vecinos)))
    X_train = train['X']
    y_train = train['Y']
    index =  train['index']
    X_test = test['X']
    y_test = test['Y'] 
    print(y_train)
    print(list(set(y_train)))
    print(index)
    
    print("Model with dim="+str(dim)+", vecinos="+str(vecinos)+" y C="+str(c))
    clf = LogisticRegression(C=c,max_iter=1000000)
    clf.fit(preprocessing.scale(X_train), y_train)
    y_pred = clf.predict(preprocessing.scale(X_test))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    
    filename = 'lr_%s_%s.pckl' % (str(dim), str(c))
    elapsed_time = perf_counter() - t0
    print("Time "+str(elapsed_time))

    pickle.dump(clf, open(filename,'wb'), protocol=4)

    result ={"Dimension":dim,"Vecinos":vecinos,"C":c,"Accuracy":accuracy_score(y_test, y_pred)}
    with open("lriso.csv","a+", newline='') as file:
            dict_writer = DictWriter(file, fieldnames=field_names)
            dict_writer.writerow(result)
