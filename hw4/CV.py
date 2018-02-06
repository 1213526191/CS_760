from bayes import *
import random
import numpy as np
from scipy.stats import ttest_ind


def setCVIndex(K, n):
    a = range(n)
    random.shuffle(a)
    k = int(round(1.*n/K))
    index = [0 for i in range(n)]
    for i in range(K-1):
        for j in range((i*k),(i*k+k)):
            index[j] = i+1
    CVIndex_dir = {a[i]: index[i] for i in range(n)}
    CVIndex = [CVIndex_dir[i] for i in range(n)]
    return CVIndex





K = 10

fname = 'chess-KingRookVKingPawn.arff.txt'
X, Y, metadata, feature_range = loadData(fname)
n = len(Y)
CVIndex = setCVIndex(K, n)
CVIndex = np.array(CVIndex)
accuracy = np.zeros((K,2))
for i in range(K):
    print i
    index_test = CVIndex == i
    index_train = CVIndex != i
    X_train, Y_train = X[index_train], Y[index_train]
    X_test, Y_test = X[index_test], Y[index_test]
    # fit Naive Bayes
    P_Y, P_XgY = bayesDist(X_train, Y_train, feature_range)
    Y_hat, Y_prob = navieBayesPrediction(X_test, P_Y, P_XgY, feature_range)
    accuracy[i, 0] = 1.0 * sum(Y_hat == Y_test) / len(Y_test)
    # fit TAN
    P_Y, P_XgY = bayesDist(X_train, Y_train, feature_range)
    parent = tanStructure(X_train, Y_train, metadata, feature_range)
    P_XgXY = computeP_XgXY(X_train, Y_train, feature_range, parent)
    Y_hat, Y_prob = computePredictions_TAN(X_test, P_XgXY, parent, feature_range)
    accuracy[i, 1] = 1.0 * sum(Y_hat == Y_test) / len(Y_test)
print(accuracy)

t, p = ttest_ind(accuracy[:, 0], accuracy[:, 1], equal_var=False)
print("ttest_ind:            t = %g  p = %g" % (t, p))
