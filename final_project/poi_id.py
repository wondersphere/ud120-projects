#!/usr/bin/python

import sys
import pickle
# sys.path.append("../tools/")
import os

from sklearn.preprocessing.data import StandardScaler
sys.path.append(os.path.join(os.getcwd(), "tools"))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary'] # You will need to use more features
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
    'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
features_list = ['poi'] + financial_features + email_features

### Load the dictionary containing the dataset
# with open("final_project_dataset.pkl", "r") as data_file:
with open(os.path.join(os.getcwd(), "final_project", "final_project_dataset.pkl"), "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

# from enron_outliers.py
data_dict.pop("TOTAL", 0)
data_dict.pop("LAY KENNETH L",0)
data_dict.pop("SKILLING JEFFREY K",0)
data_dict.pop("LAVORATO JOHN J",0)
data_dict.pop("FREVERT MARK A",0)


### Task 3: Create new feature(s)

# Add "fraction_from_poi" and "fraction_to_poi" features
for name in data_dict:
    try:
        data_dict[name]["fraction_from_poi"] = 1. * data_dict[name]["from_poi_to_this_person"] / data_dict[name]["from_messages"]
    except:
        data_dict[name]["fraction_from_poi"] = 0.
    try:
        data_dict[name]["fraction_to_poi"] = 1. * data_dict[name]["from_this_person_to_poi"] / data_dict[name]["to_messages"]
    except:
        data_dict[name]["fraction_to_poi"] = 0.
features_list = features_list + ["fraction_from_poi", "fraction_to_poi"]
print(features_list)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = False)
labels, features = targetFeatureSplit(data)

# Import only a select number of features (num_features = 10) using SelectKBest

from sklearn.feature_selection import SelectKBest
num_features = 10
selection = SelectKBest()
selection.fit(features, labels)
best_features = list(zip(features_list[1:], selection.scores_))
best_features.sort(key = lambda feat: feat[1], reverse= True)
select_features = [i[0] for i in best_features[:num_features]]

# Redefine features_list using the result from SelectKBest
features_list = ['poi'] + select_features

# Recreate the dataset so it only contains the selected features
data = featureFormat(my_dataset, features_list, sort_keys = False)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Scale the values
from sklearn.preprocessing import StandardScaler

mmscaler = StandardScaler()
mmscaler.fit(features_train)
features_train = mmscaler.transform(features_train)
features_test = mmscaler.transform(features_test)
print("Scaling done")

# Fit the model
clf.fit(features_train, labels_train)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

# # Result
#         Accuracy: 0.84014       Precision: 0.43815      Recall: 0.42150 F1: 0.42966     F2: 0.42473
#         Total predictions: 14000        True positives:  843    False positives: 1081   False negatives: 1157   True negatives: 10919