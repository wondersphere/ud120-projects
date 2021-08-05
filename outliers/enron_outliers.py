#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
# sys.path.append("../tools/")
import os
sys.path.append(os.path.join(os.getcwd(), "tools"))

from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
file_path = os.path.join(os.getcwd(), "final_project", "final_project_dataset.pkl")

# data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict = pickle.load( open(file_path, "r") )

# There's one outlier that should pop out to you immediately. Now the question is to identify the source.
# We found the original data source to be very helpful for this identification; you can find that PDF in
# final_project/enron61702insiderpay.pdf
# What's the name of the dictionary key of this data point? (e.g. if this is Ken Lay, the answer would be "LAY KENNETH L").
max_bonus = 0
max_bonus_name = ""
for name in data_dict:
    try:
        if int(data_dict[name]["bonus"]) > max_bonus:
            max_bonus = data_dict[name]["bonus"]
            max_bonus_name = name
    except:
        pass
print("Outlier bonus:", max_bonus)
# ('Outlier bonus:', 97343619)
print("Key for outlier bonus:", max_bonus_name)
# ('Key for outlier bonus:', 'TOTAL')
# It's the TOTAL row that got included in the dataset

# Remove the "TOTAL" from the dataset
data_dict.pop("TOTAL", 0)

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



# We would argue that there's 4 more outliers to investigate; let's look at a couple of them. Two people made bonuses of at
# least 5 million dollars, and a salary of over 1 million dollars; in other words, they made out like bandits. What are the
# names associated with those points?
# 2 people with > 5e6 bonuses and > 1e6 salaries
name_outlier = []
for name in data_dict:
    try:
        if (int(data_dict[name]["salary"]) > 1e+6) and (int(data_dict[name]["bonus"]) > 5e+6):
            name_outlier.append(name)
    except:
        pass
print("Leftover Outlier:", name_outlier)
# ('Leftover Outlier:', ['LAY KENNETH L', 'SKILLING JEFFREY K'])
# 4 outlier people
name_outlier = []
for name in data_dict:
    try:
        if (int(data_dict[name]["salary"]) > 1e+6) or (int(data_dict[name]["bonus"]) > 7e+6):
            name_outlier.append(name)
    except:
        pass
print("Leftover Outlier:", name_outlier)
# ('Leftover Outlier:', ['LAVORATO JOHN J', 'LAY KENNETH L', 'SKILLING JEFFREY K', 'FREVERT MARK A'])