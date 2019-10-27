""" We simulate the situation that we have a small amount of new data coming in. 
Every time we call read_data  it return a randomized, small fraction of the digits dataset. """
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict, KFold
import numpy
import pickle
import pandas as pd

import os

def load_digits_biased(remove_digits):

    training_data = numpy.array(pd.read_csv("data_classification/mnist_train.csv"))
    X = training_data[:,1::]
    y = training_data[:,0]
     
    if remove_digits:
        delete_rows = []
        for digit, amount in remove_digits.items():
            idx = numpy.where(y==digit)[0]
            delete_rows += list(idx[0:int(len(idx)*amount)])
    
        X = numpy.delete(X,delete_rows,0)
        y = numpy.delete(y,delete_rows,0)

    return {'data':X, 'target':y}
    
def read_training_data(remove_digits=None, sample_fraction=1.0):
    dataset = load_digits_biased(remove_digits)
    x  = dataset['data']
    y  = dataset['target']
    if sample_fraction < 1.0:
        foo, x, bar, y = train_test_split(x, y, test_size=sample_fraction)
    classes = range(10)
    return (x, y, classes)


def read_test_data():

    test_data = numpy.array(pd.read_csv("data_classification/mnist_test.csv"))
    X_validate = test_data[:,1::]
    y_validate = test_data[:,0]

    classes = range(10)
    return (X_validate, y_validate, classes)
