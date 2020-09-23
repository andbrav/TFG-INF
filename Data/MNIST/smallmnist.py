import os
import numpy as np
from tensorflow.keras.datasets import mnist
current_dir = os.getcwd()
(X_train, y_train), (X_test, y_test) = mnist.load_data(path=current_dir+'/mnist.npz')
sample = np.random.choice(X_train.shape[0], size=10000, replace=False)
X_small=X_train[sample, :]
y_small=y_train[sample]
print(X_small)
print(X_small.shape)
print(y_small)
print(y_small.shape)
np.savez_compressed('mnist_small.npz',X=X_small,y=y_small)

mnist= np.load('mnist_small.npz')
print(mnist)
X_small=mnist['X']
y_small=mnist['y']



unique, counts = np.unique(y_small, return_counts=True)
print(dict(zip(unique, counts)))
