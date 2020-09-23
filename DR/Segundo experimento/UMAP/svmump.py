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
from time import perf_counter


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
(X_orig, y_orig), (_,_) = mnist.load_data(path=current_dir+'/../../../Data/MNIST/mnist.npz')


#Parameters
dr= [(1,500,0.5),(2,5,0.25),(3,5,0.75),(4,5,0.99),(5,1000,0.99),(6,5,0.99),(7,5,0.5),(8,5,0.99),(9,5,0.75),(10,5,0.75),(50,5,0.1),(100,5,0.75),(200,5,0.75)]

dataset = current_dir+"/../../../Data/MNIST/mnist_mediumsmall.npz"
data = np.load(dataset)
index = data['index']
index = np.append(index,np.array([i for i in range(60000) if i not in index]))

field_names = ["Dimension","Accuracy_standard","Accuracy_vsvm"]
with open("svmumap.csv","w", newline='') as file:
    filewriter = writer(file, delimiter=',')
    filewriter.writerow(field_names)


for d in dr:
    t0 = perf_counter()
    dim = d[0]
    vecinos = d[1]
    dist = d[2]
    print(str(dim)+" "+ str(vecinos)+" "+str(dist))
    drmodel = pickle.load(open('umap%s.0_%s_%s.pckl' % (str(dim),str(vecinos),str(dist)),"rb"))
    train = np.load('umaptrain%s.0_%s_%s.npy.npz'  % (str(dim),str(vecinos),str(dist)))
    test = np.load('umaptest%s.0_%s_%s.npy.npz'  % (str(dim),str(vecinos),str(dist)))
    X_train = train['X']
    y_train = train['Y']
    index =  train['index']
    X_test = test['X']
    y_test = test['Y'] 
    print(y_train)
    print(list(set(y_train)))
    print(index)
    
    print("Model with dim="+str(dim)+" y vecinos="+str(vecinos))
    svclassifier = SVC()
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ac_st = accuracy_score(y_test, y_pred)

    # Virtual data creation
    print("Number of support points "+str(svclassifier.support_.shape[0]))
    print(X_orig[index[svclassifier.support_]].shape)

    VX_train = np.array([image for SV in X_orig[index[svclassifier.support_]] for image in onepixeltranslations(SV)])
    Vy_train = np.array([label for label in  y_orig[index[svclassifier.support_]] for i in range(8)])
    print(VX_train.shape)
    print(Vy_train.shape)


    nsamples, nx, ny = VX_train.shape
    VX_train = VX_train.reshape((nsamples,nx*ny))
    VX_train = drmodel.transform(VX_train)



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
    filename = 'svm_base_%s_%s_%s.pckl' % (str(dim),str(vecinos),str(dist))
    elapsed_time = perf_counter() - t0
    print("Time "+str(elapsed_time))

    pickle.dump(svclassifier2, open(filename,'wb'), protocol=4)
    result ={"Dimension":dim,"Accuracy_standard":ac_st,"Accuracy_vsvm":ac_svm}
    with open("svmumap.csv","a+", newline='') as file:
            dict_writer = DictWriter(file, fieldnames=field_names)
            dict_writer.writerow(result)







