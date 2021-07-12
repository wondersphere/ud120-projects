#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gnb_classifier = GaussianNB()

t0 = time()
gnb_classifier.fit(features_train, labels_train)
t_train = round(time()-t0, 3)

t0 = time()
pred = gnb_classifier.predict(features_test)
t_predict = round(time()-t0, 3)

accuracy = accuracy_score(labels_test, pred)
print(accuracy)

print "training time:", t_train, "s"
print "prediction time:", t_predict, "s"


#########################################################


