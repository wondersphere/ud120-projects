#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
#sys.path.append("../tools/")

import os
sys.path.append(os.path.join(os.getcwd(), "tools"))

from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print("Train Data:",len(features_train), len(labels_train))
print("Test Data:", len(features_test), len(labels_test))

#########################################################
### your code goes here ###

from sklearn import svm
from sklearn.metrics import accuracy_score

clf = svm.SVC()

t0 = time()
clf.fit(features_train, labels_train)
t_train = round(time()-t0, 3)

t0 = time()
pred = clf.predict(features_test)
t_predict = round(time()-t0, 3)

accuracy = accuracy_score(labels_test, pred)
print("Accuracy = ", accuracy)

print "training time:", t_train, "s"
print "prediction time:", t_predict, "s"

#########################################################


