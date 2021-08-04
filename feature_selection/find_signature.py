#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
# words_file = "../text_learning/your_word_data.pkl" 
# authors_file = "../text_learning/your_email_authors.pkl"
import os
words_file = os.path.join(os.getcwd(),"your_word_data.pkl") 
authors_file = os.path.join(os.getcwd(), "your_email_authors.pkl")
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_class = DecisionTreeClassifier()
dt_class.fit(features_train, labels_train)

# What's the accuracy of the decision tree you just made? (Remember, we're setting up our decision tree to overfit -- 
# ideally, we want to see the test accuracy as relatively low.)
print("Test Accuracy:", accuracy_score(labels_test, dt_class.predict(features_test)))
# ('Test Accuracy:', 0.94766780432309439)


# Take your (overfit) decision tree and use the feature_importances_ attribute to get a list of the relative importance
# of all the features being used. We suggest iterating through this list (it's long, since this is text data) and only
# printing out the feature importance if it's above some threshold (say, 0.2--remember, if all words were equally
# important, each one would give an importance of far less than 0.01). What's the importance of the most important
# feature? What is the number of this feature?
feat_importance = dt_class.feature_importances_
feat_gt_02 = []
for idx, importance in enumerate(feat_importance):
    if importance > 0.2:
        feat_gt_02.append([idx, importance])
print(feat_gt_02)
# [[33614, 0.76470588235294124]]

# In order to figure out what words are causing the problem, you need to go back to the TfIdf and use the feature numbers
# that you obtained in the previous part of the mini-project to get the associated words. You can return a list of all
# the words in the TfIdf by calling get_feature_names() on it; pull out the word that's causing most of the discrimination
# of the decision tree. What is it? Does it make sense as a word that's uniquely tied to either Chris Germany or Sara
# Shackleton, a signature of sorts?
print(vectorizer.get_feature_names()[feat_gt_02[0][0]])
# sshacklensf

# This word seems like an outlier in a certain sense, so let's remove it and refit. Go back to text_learning/vectorize_text.py, 
# and remove this word from the emails using the same method you used to remove "sara", "chris", etc. Rerun vectorize_text.py,
# and once that finishes, rerun find_signature.py. Any other outliers pop up? What word is it? Seem like a signature-type word?
# (Define an outlier as a feature with importance > 0.2, as before).

# cgermannsf