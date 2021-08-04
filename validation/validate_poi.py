#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
# sys.path.append("../tools/")
import os
sys.path.append(os.path.join(os.getcwd(),"tools"))
from feature_format import featureFormat, targetFeatureSplit

# data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )
data_dict = pickle.load(open(os.path.join(os.getcwd(),"final_project","final_project_dataset.pkl"), "r"))

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

tree_clf = DecisionTreeClassifier()

# Without train_test_split
tree_clf.fit(features, labels)
tree_clf_pred = tree_clf.predict(features)
print("Without train_test_split")
print("Accuracy:", accuracy_score(labels, tree_clf_pred))

# With train_test_split
feat_train, feat_test, labels_train, labels_test = train_test_split(
    features, labels, test_size = 0.3, random_state = 42)
tree_clf.fit(feat_train, labels_train)
tree_clf_pred = tree_clf.predict(feat_test)
print("With train_test_split")
print("Accuracy:", accuracy_score(labels_test, tree_clf_pred))