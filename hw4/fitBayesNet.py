from bayes import *

data_train, data_test, option = loadDataName()

# data_data = "lymph_train.arff"
# data_test = "lymph_test.arff"

X_train, Y_train, metadata, feature_range = loadData(data_train)
X_test, Y_test, _, _ = loadData(data_test)

if option is "n":
    P_Y, P_XgY = bayesDist(X_train, Y_train, feature_range)
    Y_hat, Y_prob = navieBayesPrediction(X_test, P_Y, P_XgY, feature_range)
    print1(metadata)
    printTestResults(Y_hat, Y_prob, Y_test, metadata)



## ---- TAN----


if option is "t":
# data_data = "lymph_train.arff"
# data_test = "lymph_test.arff"

# X_train, Y_train, metadata, feature_range = loadData(data_data)
# X_test, Y_test, _, _ = loadData(data_test)
    P_Y, P_XgY = bayesDist(X_train, Y_train, feature_range)
    parent = tanStructure(X_train, Y_train, metadata, feature_range)
    printGraph_TAN(parent, metadata)
    P_XgXY = computeP_XgXY(X_train, Y_train, feature_range, parent)
    Y_hat, Y_prob = computePredictions_TAN(X_test, P_XgXY, parent, feature_range)
    printTestResults(Y_hat, Y_prob, Y_test, metadata)