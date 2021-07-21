#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    error = map(lambda x, y: abs(x - y), predictions, net_worths)
    cleaned_data = sorted(zip(ages, net_worths, error), key = lambda x: x[2])[:-10]
 

    return cleaned_data

