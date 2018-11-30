
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import time


# In[2]:


X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy') 
#
Y_train = np.load('Y_train.npy') 
#
Y_test = np.load('Y_test.npy') 
#
#, X_test, Y_train, Y_test
print(np.unique(Y_test).shape[0])


# In[3]:


# C=1.0
# models = (svm.SVC(kernel='linear', C=C),
#           svm.SVC(kernel='rbf', gamma=0.7, C=C),
#           svm.SVC(kernel='poly', degree=3, C=C))
# models = (clf.fit(X_train, Y_train) for clf in models)


# In[6]:


from sklearn.svm import SVC 
C = 1.0
svm_model_linear = (svm.SVC(kernel='linear', C=C,decision_function_shape='ovr'),
          svm.SVC(kernel='rbf', gamma=0.7, C=C,decision_function_shape='ovr'),
          svm.SVC(kernel='poly', degree=2, C=C,decision_function_shape='ovr'))
titles = ('SVC with linear kernel',
           'SVC with RBF kernel',
          'SVC with polynomial (degree 2) kernel')
models = (clf.fit(X_train, Y_train.ravel()) for clf in svm_model_linear)

for svm_model_linear,title in zip(models,titles):
    tic = time.clock()
    svm_predictions = svm_model_linear.predict(X_test) 
    print(title)
    print('No of support vectors per class : '+ str(svm_model_linear.support_vectors_.shape[0]))
    print('No of class : '+ str(svm_model_linear.classes_.shape[0]))
    toc = time.clock()
    print('time per prediction : ' + str((toc-tic)/X_test.shape[0]))
    accuracy = svm_model_linear.score(X_test, Y_test.ravel()) 
    print('accuracy : ' + str(accuracy*100) +' %\n\n')

