# coding=utf-8
#########################################
# kNN: k Nearest Neighbors  

# Input:      newInput: vector to compare to existing dataset (1xN)  
#             dataSet:  size m data set of known vectors (NxM)  
#             labels:   data set labels (1xM vector)  
#             k:        number of neighbors to use for comparison   

# Output:     the most popular class label  
#########################################  

from numpy import *
import operator


def read_DataSet():
    # todo:特征标准化..
    labels = genfromtxt('data.csv', delimiter=',', dtype='str', skip_header=1, usecols=(1))
    group = genfromtxt('data.csv', delimiter=',', skip_header=1)
    train_data = group[:400]
    train_labels = labels[:400]
    test_data = group[400:]
    test_labels = labels[400:]
    return delete(train_data, [0, 1], axis=1), train_labels, delete(test_data, [0, 1], axis=1), test_labels


# create a dataset which contains 4 samples with 2 classes
def createDataSet():
    # create a matrix: each row as a sample  
    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']  # four samples and two classes
    return group, labels


# classify using kNN  
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row

    ## step 1: calculate Euclidean distance  
    # tile(A, reps): Construct an array by repeating A reps times  
    # the following copy numSamples rows for dataSet  
    diff = tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    ## step 2: sort the distance  
    # argsort() returns the indices that would sort an array in a ascending order  
    sortedDistIndices = argsort(distance)

    classCount = {}  # define a dictionary (can be append element)
    for i in xrange(k):
        ## step 3: choose the min k distance  
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur  
        # when the key voteLabel is not in dictionary classCount, get()  
        # will return 0  
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        ## step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex


def compare(y, real_y):
    """
    对比预测和真实结果
    :param y: 预测的分类结果
    :param real_y: 真实的分类结果
    :return: 准确率
    """
    n = len(y)
    s = 0
    for i in xrange(n - 1):
        if y[i] == real_y[i]:
            s += 1
    return 1.0 * s / n


if __name__ == '__main__':
    dataSet, labels, test_data, test_labels = read_DataSet()

    k = 5
    plabels = []
    for testX in test_data:
        outputLabel = kNNClassify(testX, dataSet, labels, k)
        plabels.append(outputLabel)
        # print "Your input is:", testX, "and classified to class: ", outputLabel
    print(plabels)
    print(compare(plabels, test_labels))

