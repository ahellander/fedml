import numpy as np
from random import sample
from sklearn.model_selection import KFold
#import reg_dataset_preprocessing as data
import classification_dataset_preprocessing as data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets.samples_generator import make_blobs
import random


# disjoint partition without resampling
def equal_partition(n_train, N=2):
    # shuffle the index first
    randIndex = sample(list(range(n_train)), n_train)
    splitLen = n_train//N
    #print(n_train, N, splitLen)
    list_part = np.zeros((N, splitLen))
    list_part[0, :] = randIndex[0:splitLen]
    for i in range(1, N):
        startIndex = i*splitLen
        list_part[i, :] = randIndex[startIndex:(startIndex+splitLen)]

    list_part = list_part.astype(int)

    return list_part

# disjoint partition without resampling
def stratified_equal_partition(n_train, y, N=2):
    # shuffle the index first
    trainIndex = list(range(n_train))
    splitLen = n_train // N
    ss = StratifiedShuffleSplit(n_splits=N, test_size=splitLen)
    list_part = np.zeros((N, splitLen))
    indx = 0
    for train_index, test_index in ss.split(trainIndex, y):
        list_part[indx, :] = test_index
        indx+= 1

    list_part = list_part.astype(int)

    return list_part


# returns equal partition with overlapping examples,
# the size of the partition can also be mentioned
def equal_partition_overlap(n_train, N=2, size = None):
    if size is None:
        size = n_train // N

    list_part = np.zeros((N, size))

    for i in range(N):
        # shuffle the index first
        randIndex = sample(list(range(n_train)), n_train)
        trainIndex = sample(randIndex, size)
        list_part[i, :] = trainIndex

    list_part = list_part.astype(int)

    return list_part


# might overlap
def partition_with_diffSizes(n_train, sampleRatio):
    N = len(sampleRatio)
    list_part = list()

    for i in range(N):
        trainIndex = sample(list(range(n_train)), int(sampleRatio[i]*n_train))
        list_part.append(trainIndex)

    return list_part


# disjoint partition without resampling
def unbalanced_digits_partition(y_train, N=2):
    # shuffle the index first
    n_train = len(y_train)
    classes = np.unique(y_train)
    splitLen = y_train//N
    temp_cls = len(classes)//2
    train_index = np.array(range(n_train))
    #train_index = list(range(n_train))

    data_classes = [[], [], [], [], []]

    # separate into 3 groups
    subset1 = y_train == classes[0]
    subset2 = y_train == classes[5]
    ind = subset1 | subset2
    data_classes[0] = train_index[ind]
    subset1 = y_train == classes[1]
    subset2 = y_train == classes[6]
    ind = subset1 | subset2
    data_classes[1] = train_index[ind]
    subset1 = y_train == classes[2]
    subset2 = y_train == classes[7]
    ind = subset1 | subset2
    data_classes[2] = train_index[ind]

    subset1 = y_train == classes[3]
    subset2 = y_train == classes[8]
    ind = subset1 | subset2
    data_classes[3] = train_index[ind]

    subset1 = y_train == classes[4]
    subset2 = y_train == classes[9]
    ind = subset1 | subset2
    data_classes[4] = train_index[ind]

    list_part = list()

    # Please make sure N is a multiple of len(classes)/2
    #print(n_train, N, splitLen)
    for i in range(temp_cls):
        #randIndex = sample(data_classes[i], len(data_classes[i]))
        random.shuffle(data_classes[i])
        randIndex = data_classes[i].astype(int)
        splitLen = int((temp_cls / N) * len(data_classes[i]))
        list_part.append(randIndex[0:splitLen])
        for j in range(1, N//temp_cls):
            startIndex = j * splitLen
            list_part.append(randIndex[startIndex:(startIndex + splitLen)])

    #list_part = list_part.astype(int)

    return list_part

# disjoint partition without resampling
def unbalanced_partition(y_train, N=2):
    # shuffle the index first
    n_train = len(y_train)
    classes = np.unique(y_train)
    splitLen = y_train//N
    temp_cls = len(classes)//2
    train_index = np.array(range(n_train))
    #train_index = list(range(n_train))

    data_classes = [[], [], [], [], []]

    # separate into 3 groups
    subset1 = y_train == classes[0]
    subset2 = y_train == classes[2]
    ind = subset1 | subset2
    data_classes[0] = train_index[ind]
    subset1 = y_train == classes[1]
    subset2 = y_train == classes[3]
    ind = subset1 | subset2
    data_classes[1] = train_index[ind]
    subset1 = y_train == classes[4]
    subset2 = y_train == classes[5]
    ind = subset1 | subset2
    data_classes[2] = train_index[ind]

    list_part = list()

    # Please make sure N is a multiple of len(classes)/2
    #print(n_train, N, splitLen)
    for i in range(temp_cls):
        #randIndex = sample(data_classes[i], len(data_classes[i]))
        random.shuffle(data_classes[i])
        randIndex = data_classes[i].astype(int)
        splitLen = int((temp_cls / N) * len(data_classes[i]))
        list_part.append(randIndex[0:splitLen])
        for j in range(1, N//temp_cls):
            startIndex = j * splitLen
            list_part.append(randIndex[startIndex:(startIndex + splitLen)])

    #list_part = list_part.astype(int)

    return list_part


if __name__=="__main__":
    X, y = data.load_digit_data()
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                        random_state=3, stratify=y)

    N = 6
    list_part = unbalanced_digits_partition(y_train, N)
    for part_index in list_part:
        x_part = X_train[part_index]
        y_part = y_train[part_index]
        # print(part_index)
        print(y_part)
        # print(x_part.shape)


    '''
    X, y = data.load_covertype_dataset()
    #X, y = make_blobs(n_samples=300, centers=6, n_features=2)
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                        random_state=3, stratify=y)

    N = 6
    list_part = unbalanced_partition(X_train, y_train, N)
    for part_index in list_part:
        x_part = X_train[part_index]
        y_part = y_train[part_index]
        #print(part_index)
        print(y_part)
        #print(x_part.shape)

    '''

    '''
    X, y = data.load_spambase_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                        random_state=3, stratify=y)

    N = 3
    print(len(X_train))
    list_part =  stratified_equal_partition(len(X_train), y_train, N)
    for part_index in list_part:
        x_part = X_train[part_index]
        y_part = y_train[part_index]
        print(x_part.shape)

    
    size = [.4, .5, .8]
    # partition train data only, keeping the test data aside
    list_part = partition_with_diffSizes(len(X_train), size)
    # print(list_part)
    for part_index in list_part:
        x_part =  X_train[part_index]
        y_part = y_train[part_index]
        print(x_part.shape)


    list_part = equal_partition(len(X_train), N=3)
    #list_part = equal_partition_overlap(len(X_train), N, size=2000)
    # print(list_part)
    for part_index in list_part:
        x_part =  X_train[part_index]
        y_part = y_train[part_index]
        print(part_index)
    '''
