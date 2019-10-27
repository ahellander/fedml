"""
 Author: Niharika gauraha
 Preprocess classification datasets
"""

import csv
from sklearn.datasets import load_breast_cancer, load_digits
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_breast_cancer_data():
    bc = load_breast_cancer()
    X = bc.data
    y = bc.target

    return X, y

def load_digit_data():
    digit = load_digits()
    X = digit.data
    y = digit.target

    return X, y

def load_spambase_data():
    data = []
    # Read the training data
    file = open('data_classification/spambase.data')
    reader = csv.reader(file)

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.int)
    del data # free up the memory

    return X, y


def load_Phishing_dataset():
    data = []
    # Read the training data
    file = open('data_classification/phising.csv')
    reader = csv.reader(file, delimiter = ',')

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.int)
    y[y==-1] = 0
    del data # free up the memory

    return X, y


# generic function, can be used with any dataset file
def load_dataset(filename):
    data = []
    # Read the training data
    file = open(filename)
    reader = csv.reader(file, delimiter = ' ')

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.int)
    del data # free up the memory

    return X, y


def load_bank_dataset():
    data = []
    # Read the training data
    file = open('data_classification/bank.csv')
    reader = csv.reader(file, delimiter = ',')

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data])
    y = np.array([x[-1] for x in data])
    del data # free up the memory

    le = LabelEncoder()
    for i in range(X.shape[1]):
        try:
            X[:, i] = X[:, i].astype(np.float)
        except:
            le.fit(X[:, i])
            X[:, i] = le.transform(X[:, i])

    y[y == 'no'] = 0
    y[y == 'yes'] = 1
    y = y.astype(np.int)

    return X, y

def load_australian_dataset():
    data = []
    # Read the training data
    file = open('data_classification/australian.dat')
    reader = csv.reader(file, delimiter = ' ')

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data])
    y = np.array([x[-1] for x in data])
    del data # free up the memory

    le = LabelEncoder()
    for i in range(X.shape[1]):
        try:
            X[:, i] = X[:, i].astype(np.float)
        except:
            le.fit(X[:, i])
            X[:, i] = le.transform(X[:, i])

    y = y.astype(np.int)
    y[y == -1] = 0

    return X, y

def load_adult_dataset():
    data = []
    # Read the training data
    file = open('data_classification/adult.data')
    reader = csv.reader(file, delimiter = ',')

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data])
    y = np.array([x[-1] for x in data])
    del data # free up the memory

    le = LabelEncoder()
    for i in range(X.shape[1]):
        try:
            X[:, i] = X[:, i].astype(np.float)
        except:
            le.fit(X[:, i])
            X[:, i] = le.transform(X[:, i])

    y = y.astype(np.int)
    y[y == -1] = 0

    return X, y


def load_tic_tac_toe_dataset():
    data = []
    # Read the training data
    file = open('data_classification/tic-tac-toe.data')
    reader = csv.reader(file, delimiter=',')

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data])
    y = np.array([x[-1] for x in data])
    del data # free up the memory

    le = LabelEncoder()
    for i in range(X.shape[1]):
        try:
            X[:, i] = X[:, i].astype(np.float)
        except:
            le.fit(X[:, i])
            X[:, i] = le.transform(X[:, i])

    y[y == 'negative'] = 0
    y[y == 'positive'] = 1
    y = y.astype(np.int)

    return X, y


def load_covertype_dataset():
    data = []
    # Read the training data
    file = open('data_classification/covtype.data')
    reader = csv.reader(file, delimiter = ',')

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.int)
    y = y-1
    del data # free up the memory
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1-(1875/len(X)), stratify=y)
    # it is huge dataset, returning only slice of it
    return x_train, y_train


def load_monks1_data():
    data = []
    # Read the training data
    file = open('data_classification/monks-1.train')
    reader = csv.reader(file, delimiter = ' ')

    for row in reader:
        data.append(row)
    file.close()
    #skip one '' string read from file
    X1 = np.array([x[2:-1] for x in data]).astype(np.float)
    y1 = np.array([x[1] for x in data]).astype(np.float)
    del data  # free up the memory
    data = []
    file = open('data_classification/monks-1.test')
    reader = csv.reader(file, delimiter=' ')

    for row in reader:
        data.append(row)
    file.close()
    X2 = np.array([x[2:-1] for x in data]).astype(np.float)
    y2 = np.array([x[1] for x in data]).astype(np.int)

    X = np.row_stack((X1,X2))
    y = np.concatenate((y1,y2))
    y = y.astype(np.int)

    return X, y


def load_monks2_data():
    data = []
    # Read the training data
    file = open('data_classification/monks-2.train')
    reader = csv.reader(file, delimiter = ' ')

    for row in reader:
        data.append(row)
    file.close()
    #skip one '' string read from file
    X1 = np.array([x[2:-1] for x in data]).astype(np.float)
    y1 = np.array([x[1] for x in data]).astype(np.float)
    del data  # free up the memory
    data = []
    file = open('data_classification/monks-2.test')
    reader = csv.reader(file, delimiter=' ')

    for row in reader:
        data.append(row)
    file.close()
    X2 = np.array([x[2:-1] for x in data]).astype(np.float)
    y2 = np.array([x[1] for x in data]).astype(np.int)

    X = np.row_stack((X1,X2))
    y = np.concatenate((y1,y2))
    y = y.astype(np.int)

    return X, y


if __name__ == '__main__':
    x, y = load_spambase_data()
    #x, y = load_breast_cancer_data()
    #x, y = load_Phishing_dataset()
    #x, y = load_covertype_dataset()
    #x, y = load_adult_dataset()
    #x, y = load_tic_tac_toe_dataset()
    #x, y = load_australian_dataset()
    #x, y = load_monks1_data()
    #x, y = load_monks2_data()
    #x, y = load_bank_dataset()
    #x, y = load_digit_data()

    print(len(y[y==0]))
    print(len(y[y == 1]))


