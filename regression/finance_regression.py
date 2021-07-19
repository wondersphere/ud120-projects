#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    


import sys
import pickle
# sys.path.append("../tools/")
import os
sys.path.append(os.path.join(os.getcwd(), "tools"))
from feature_format import featureFormat, targetFeatureSplit
# dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )
file_path = os.path.join(os.getcwd(), "final_project", "final_project_dataset.pkl")
dictionary = pickle.load(open(file_path, "rb"))

### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]

# data = featureFormat( dictionary, features_list, remove_any_zeroes=True)

# From Python 3.3 forward, a change to the order in which dictionary keys are processed was made such that the orders are randomized
# each time the code is run. This will cause some compatibility problems with the graders and project code, which were run under
# Python 2.7. To correct for this, add the following argument to the featureFormat call on line 26 of finance_regression.py:
#   sort_keys = '../tools/python2_lesson06_keys.pkl'
# This will open up a file in the tools folder with the Python 2 key order.

sort_key_path = os.path.join(os.getcwd(), "tools", "python2_lesson06_keys.pkl")
data = featureFormat( dictionary, features_list, remove_any_zeroes=True, sort_keys = sort_key_path)

target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(feature_train, target_train)

print("Slope:",reg.coef_)
print("Intercept:", reg.intercept_)

# Imagine you were a less savvy machine learner, and didn't know to test on a holdout test set. Instead, you tested on the same data that you
# used to train, by comparing the regression predictions to the target values (i.e. bonuses) in the training data. What score do you find?
# You may not have an intuition yet for what a "good" score is, this score isn't very good (but it can be a lot worse).
print("Score_Train:", reg.score(feature_train, target_train))

# Now compute the score for your regression on the test data, like you know you should. What's that score on the testing data? If you made the
# mistake of only assessing on the training data, would you overestimate or underestimate the performance of your regression?
print("Score_Test:", reg.score(feature_test, target_test))

# There are lots of finance features available, some of which might be more powerful than others in terms of predicting a person's bonus. For
# example, suppose you thought about the data a bit and guess that the "long_term_incentive" feature, which is supposed to reward employees
# for contributing to the long-term health of the company, might be more closely related to a person's bonus than their salary is.
# A way to confirm that you're right in this hypothesis is to regress the bonus against the long term incentive, and see if the regression
# score is significantly higher than regressing the bonus against the salary.
# Perform the regression of bonus against long term incentive -- what's the score on the test data?

# print("long_term_incentive vs salary:")
# features_list = ["bonus", "long_term_incentive"]
# data = featureFormat( dictionary, features_list, remove_any_zeroes=True, sort_keys = sort_key_path)
# target, features = targetFeatureSplit( data )
# feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
# reg = LinearRegression()
# reg.fit(feature_train, target_train)
# print("Score_Test:", reg.score(feature_test, target_test))


### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass

# Go back to a setup where you are using the salary to predict the bonus, and rerun the code to remind yourself of what the data look like.
# You might notice a few data points that fall outside the main trend, someone who gets a high salary (over a million dollars!) but a
# relatively small bonus. This is an example of an outlier, and we'll spend lots of time on them in the next lesson.
# A point like this can have a big effect on a regression: if it falls in the training set, it can have a significant effect on the slope/
# intercept if it falls in the test set, it can make the score much lower than it would otherwise be As things stand right now, this point
# falls into the test set (and probably hurting the score on our test data as a result). Let's add a little hack to see what happens if it
# falls in the training set instead. Add these two lines near the bottom of finance_regression.py, right before plt.xlabel(features_list[1]):
#   reg.fit(feature_test, target_test)
#   plt.plot(feature_train, reg.predict(feature_train), color="b")
# Now we’ll be drawing two regression lines, one fit on the test data (with outlier) and one fit on the training data (no outlier). Look at the plot now--big difference, huh? That single outlier is driving most of the difference. What’s the slope of the new regression line?

# # With outlier in training set
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b")
print(reg.coef_)

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()


# test lineaaa