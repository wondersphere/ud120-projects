#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
# sys.path.append("../tools/")
import os
sys.path.append(os.path.join(os.getcwd(), "tools"))

from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
# data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
file_path = os.path.join(os.getcwd(), "final_project", "final_project_dataset.pkl")
data_dict = pickle.load( open(file_path, "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2, _ in finance_features:
    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

from sklearn.cluster import KMeans

cpred = KMeans(n_clusters=2)
cpred.fit(finance_features)
pred = cpred.labels_


### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters_3feat.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"

# What are the maximum and minimum values taken by the "exercised_stock_options" feature used in this example?
min_stock = 1e+20
max_stock = 0
for i in finance_features:
    try:
        if i[1] > max_stock:
            max_stock = i[1]
        if i[1] < min_stock and i[1] > 0:
            min_stock = i[1]
    except:
        pass
print("Maximum excercised_stock_options:", max_stock)
print("Minimum excercised_stock_options:", min_stock)

# What are the maximum and minimum values taken by "salary"?
max_salary = 0
min_salary = 1e+20
for name in data_dict:
    try:
        if int(data_dict[name]["salary"]) > max_salary:
            max_salary = data_dict[name]["salary"]
        if int(data_dict[name]["salary"]) < min_salary and int(data_dict[name]["salary"]) > 0:
            min_salary = data_dict[name]["salary"]
    except:
        pass
print("Maximum salary:", max_salary)
print("Minimum salary:", min_salary)

# Apply feature scaling to your k-means clustering code from the last lesson, on the "salary" and "exercised_stock_options"
# features (use only these two features). What would be the rescaled value of a "salary" feature that had an original value
# of $200,000, and an "exercised_stock_options" feature of $1 million? (Be sure to represent these numbers as floats, not 
# integers!)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(finance_features)
scaled = scaler.transform([[200000.0, 1000000.0, 0.0]])
print("Scaled salary of 200,000:", scaled[0][0])
print("Scaled excercised_stock_options of 1,000,000:", scaled[0][1])
