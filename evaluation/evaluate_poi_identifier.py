#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
# sys.path.append("../tools/")
import os
sys.path.append(os.path.join(os.getcwd(), "tools"))
from feature_format import featureFormat, targetFeatureSplit

# data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )
data_dict = pickle.load(open(os.path.join(os.getcwd(), "final_project", "final_project_dataset.pkl"), "r"))

### add more features to features_list!
features_list = ["poi", "salary"]

# data = featureFormat(data_dict, features_list)
data = featureFormat(data_dict, features_list, sort_keys = os.path.join(os.getcwd(), "tools", "python2_lesson14_keys.pkl"))
labels, features = targetFeatureSplit(data)

### your code goes here 

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

tree_clf = DecisionTreeClassifier()

feat_train, feat_test, labels_train, labels_test = train_test_split(
    features, labels, test_size = 0.3, random_state = 42)
tree_clf.fit(feat_train, labels_train)
tree_clf_pred = tree_clf.predict(feat_test)
print("With train_test_split")
print("Accuracy:", accuracy_score(labels_test, tree_clf_pred))

# How many POIs are predicted for the test set for your POI identifier?
# (Note that we said test set! We are not looking for the number of POIs in the whole dataset.)
print("Predicted POI count:", sum(tree_clf_pred))
# Answer: 4
# How many people total are in your test set?
print("Total people in test set", len(tree_clf_pred))
# Answer: 29

# If your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be?
# Answer: (29-4)/29 = 0.862

# Look at the predictions of your model and compare them to the true test labels. Do you get any true positives? 
# (In this case, we define a true positive as a case where both the actual label and the predicted label are 1)
tp_count = 0
for i in range(len(labels_test)):
    if labels_test[i] == 1.0 and tree_clf_pred[i] == 1.0:
        tp_count +=1
print("True Positive count:", tp_count)
print(labels_test)
print(tree_clf_pred)
# Answer: 0

# As you may now see, having imbalanced classes like we have in the Enron dataset (many more non-POIs than POIs)
# introduces some special challenges, namely that you can just guess the more common class label for every point,
# not a very insightful strategy, and still get pretty good accuracy!
# Precision and recall can help illuminate your performance better. Use the precision_score and recall_score
# available in sklearn.metrics to compute those quantities.
# What's the precision?
from sklearn.metrics import classification_report
print(classification_report(labels_test, tree_clf_pred))
# Answer: POI precision = 0

# What's the recall?
# (Note: you may see a message like UserWarning: The precision and recall are equal to zero for some labels. Just
# like the message says, there can be problems in computing other metrics (like the F1 score) when precision and/or
# recall are zero, and it wants to warn you when that happens.)
# Obviously this isn't a very optimized machine learning strategy (we haven’t tried any algorithms besides the
# decision tree, or tuned any parameters, or done any feature selection), and now seeing the precision and recall
# should make that much more apparent than the accuracy did.
# Answer: POI recall = 0