#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import pickle
import os
import sys
import numpy as np

file_path = os.path.join(os.getcwd(), "final_project", "final_project_dataset.pkl")
print(file_path)

# enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
enron_data = pickle.load(open(file_path, "rb"))
# enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

# View keys
# print("Keys (Names)")
# print(enron_data.keys())

# Number of data
print(" Number of data")
print(len(enron_data))

# Number of features
print("Number of features")
print(len(enron_data["SKILLING JEFFREY K"]))
# {'salary': 1111258, 'to_messages': 3627, 'deferral_payments': 'NaN', 'total_payments': 8682716, 'loan_advances': 'NaN', 'bonus': 5600000, 'email_address': 'jeff.skilling@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 26093672, 'expenses': 29336, 'from_poi_to_this_person': 88, 'exercised_stock_options': 19250000, 'from_messages': 108, 'other': 22122, 'from_this_person_to_poi': 30, 'poi': True, 'long_term_incentive': 1920000, 'shared_receipt_with_poi': 2042, 'restricted_stock': 6843672, 'director_fees': 'NaN'}

# Number of POI in dataset
print("Number of POI in dataset")
poi_sum = 0
for name in enron_data.keys():
    poi_sum += enron_data[name]["poi"]
print(poi_sum)

# Total value of stock belonging to James Prentice
print("Total value of stock belonging to James Prentice")
print(enron_data["PRENTICE JAMES"]["total_stock_value"])

# Number of emails from Wesley Colwell to POI
print("Number of emails from Wesley Colwell to POI")
print(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

# Stock value option exercised by Jeffrey K Skilling
print("Stock value option exercised by Jeffrey K Skilling")
print(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

# Total payment for Lay, Skilling, Fastow
print("Total payment for Lay, Skilling, Fastow")
print(enron_data["LAY KENNETH L"]["total_payments"])
print(enron_data["SKILLING JEFFREY K"]["total_payments"])
print(enron_data["FASTOW ANDREW S"]["total_payments"])

# How many folks in this dataset have a quantified salary? What about a known email address?
print("How many folks in this dataset have a quantified salary? What about a known email address?")
salary_count = 0
email_count = 0
for name in enron_data.keys():
    try:
        enron_data[name]["salary"] > 0
        salary_count += 1
    except:
        pass
    try:
        if enron_data[name]["email_address"].upper() != "NAN": # Apparently the NaN values are string for email addresses
            email_count += 1
    except:    
        pass
print("Salary Count:", salary_count)
print("Email Count:", email_count)

# How many people in the E+F dataset (as it currently exists) have “NaN” for their total payments? 
# What percentage of people in the dataset as a whole is this?
nan_count = 0
for name in enron_data.keys():
    if enron_data[name]["total_payments"] == "NaN": # The NaN values are string apparently
        nan_count +=1
nan_percent = nan_count/len(enron_data.keys()) * 100
print("How many people in the E+F dataset (as it currently exists) have “NaN” for their total payments?")
print(nan_count)
print("What percentage of people in the dataset as a whole is this?")
print(nan_percent, "%")

# How many POIs in the E+F dataset have “NaN” for their total payments? 
# What percentage of POI’s as a whole is this?
nan_count_poi = 0
count_poi = 0
for name in enron_data.keys():
    if enron_data[name]["poi"] == True:
        count_poi += 1
        if enron_data[name]["total_payments"] == "NaN":
            nan_count_poi +=1
nan_percent_poi = nan_count_poi/count_poi * 100
print("How many POIs in the E+F dataset have “NaN” for their total payments?")
print(nan_count_poi)
print("What percentage of POI’s as a whole is this?")
print(nan_percent_poi, "%")

# If you added in, say, 10 more data points which were all POI’s, and put “NaN” for the total payments for those folks, the numbers you just calculated would change.
# What is the new number of people of the dataset? What is the new number of folks with “NaN” for total payments?
print("If you added in, say, 10 more data points which were all POI’s, and put “NaN” for the total payments for those folks, the numbers you just calculated would change")
print("What is the new number of people of the dataset?")
print(len(enron_data.keys())+10)
print("What is the new number of folks with “NaN” for total payments")
print(nan_count+10)

# What is the new number of POI’s in the dataset? What is the new number of POI’s with NaN for total_payments?
print("What is the new number of POI’s in the dataset?")
print(count_poi+10)
print("What is the new number of POI’s with NaN for total_payments")
print(nan_count_poi+10)

# Once the new data points are added, do you think a supervised classification algorithm might interpret “NaN” for total_payments as a clue that someone is a POI?

