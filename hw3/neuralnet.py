import sys
import numpy as np
import scipy.io.arff as sparff
from random import *
import matplotlib.pyplot as plt

def loadDataName():
    dataName = str(sys.argv[1])
    nFold = int(str(sys.argv[2]))
    rate = float(str(sys.argv[3]))
    nEpoch = int(str(sys.argv[4]))
    return dataName, nFold, rate, nEpoch

def loadData(dataName):
    realName = str('./src/' + dataName)
    data, metadata = sparff.loadarff(realName)
    feature_range = []
    for name in metadata.names():
        feature_range.append(metadata[name][1])
    return data, metadata, feature_range

def cvIndex(data, labels, nFold):
    n = len(data)
    n0 = 0
    n1 = 0
    for instance in data:
        if instance[-1] == labels[0]:
            n0 += 1
        else:
            n1 += 1
    x0, x1 = [], []
    for i in range(n0):
        x0.append(randint(0, nFold - 1))
    for i in range(n1):
        x1.append(randint(0, nFold - 1))
    foldIndex = []
    i, j = 0, 0
    for instance in data:
        if instance[-1] == labels[0]:
            foldIndex.append(x0[i])
            i += 1
        else:
            foldIndex.append(x1[j])
            j += 1
    return foldIndex

def splitData(data, index, i):
    data_train = []
    data_test = []
    for j in range(len(data)):
        if index[j] == i:
            data_test.append(data[j])
        else:
            data_train.append(data[j])
    return data_train, data_test

def normalize(data):
    for i in range(len(data[0]) - 1):
        XX = []
        for j in range(len(data)):
            XX.append(data[j][i])
        XX = np.array(XX)
        Mean = np.mean(XX)
        Var = np.std(XX)
        for j in range(len(data)):
            data[j][i] = 1.*(data[j][i] - Mean)/Var
    return data

        


def cleanData(data, labels, metadata):
    data = normalize(data)
    nFeature = len(metadata.names())-1
    nInstance = len(data)
    X, Y = [], []
    for i in range(nInstance):
        if data[i][-1] == labels[0]:
            Y.append(0)
        else:
            Y.append(1)
        XX = []
        for j in range(nFeature):
            XX.append(data[i][j])
        XX.append(1)
        X.append(XX)
    return X, Y



def setWeight(nFeature, nHidden):
    weights = []
    if(nHidden == 0):
        weights.append(np.random.uniform(weight_lb, weight_up, nFeature))
    else:
        weights.append(np.random.uniform(weight_lb, weight_up, (nHidden, nFeature)))
        weights.append(np.random.uniform(weight_lb, weight_up, nHidden))
    return weights

def sigmoid(input):
    return np.divide(1.0,(np.add(1.0,np.exp(-input))))

def deltaLearn(X, Y, THRESHOLD, rate, weights):
    orders = np.random.permutation(len(Y))
    count = 0
    for i in orders:
        Input = np.array(X[i])
        output = sigmoid(np.dot(Input, weights[0]))
        delta = Y[i] - output
        weights_delta = np.multiply(rate*delta, Input)
        weights[0] += weights_delta
        if (output >= THRESHOLD and Y[i] == 1) or (output < THRESHOLD and Y[i] == 0):
            count += 1
    return weights, count 

def backprop(X, Y, THRESHOLD, rate, weights):
    orders = np.random.permutation(len(Y))
    count = 0
    for i in orders:
        Input = np.array(X[i])
        output1 = sigmoid(np.dot(weights[0], Input))
        output2 = sigmoid(np.dot(output1, weights[1]))
        delta2 = Y[i] - output2
        delta1 = delta2*weights[1]*output1*(1 - output1)
        gre1 = np.outer(delta1, Input)
        gre2 = delta2*output1
        weights[0] += gre1
        weights[1] += gre2
        if (output2 < THRESHOLD and Y[i] == 0) or (output2 >= THRESHOLD and Y[i] == 1):
            count += 1
    return weights, count 

def trainModel(data_train, THRESHOLD, labels, metadata, nHidden, rate, nEpoch):
    X, Y = cleanData(data_train, labels, metadata)
    nFeature = len(metadata.names())-1
    nInstance = len(data_train)
    weights = setWeight(len(X[0]), nHidden)
    for i in range(nEpoch):
        if(nHidden == 0):
            weights, count = deltaLearn(X, Y, THRESHOLD, rate, weights)
        else:
            weights, count = backprop(X, Y, THRESHOLD, rate, weights)
        # print ('%d\t%d\t%d' % (i, count, len(Y) - count))
    return weights
    

def testModel(data_test, THRESHOLD, labels, metadata, weights, nHidden):
    predict = []
    confidence = []
    X, Y = cleanData(data_test, labels, metadata)
    for instance in X:
        if(nHidden == 0):           
            output = sigmoid(np.dot(instance, weights[0]))
            if output < THRESHOLD:
                predict.append(0)
            else:
                predict.append(1)
        else:
            output1 = sigmoid(np.dot(weights[0], instance))
            output2 = sigmoid(np.dot(output1, weights[1]))
            if output2 < THRESHOLD:
                predict.append(0)
            else:
                predict.append(1) 
            output = output2     
        confidence .append(output)      
    return predict, confidence

def showresults(data, labels, metadata, foldIndex, predicts, confidences, show = True):
    # fold_of_instance predicted_class actual_class confidence_of_prediction 
    # index = [0 for i in range(len(predicts))]
    f = open('example.txt', 'r')
    _, Y = cleanData(data, labels, metadata)
    counts = 0
    for i in range(len(data)):
        fold_of_instance = foldIndex[i]
        predicted_class = predicts[i]
        actual_class = Y[i]
        confidence_of_prediction  = confidences[i]
        if(predicted_class == actual_class):
            counts += 1
        # index[fold_of_instance] += 1

        if show:
            print('%d\t%f\t%f\t%.4f' % (fold_of_instance, predicted_class, actual_class, confidence_of_prediction))
    if show:
        print(counts*1./len(data)) 
    return counts

def plotROC(Y, predicts, confidences):
    targ_pred = sorted(zip(Y, confidences), key=lambda x: x[1])
    targ_pred = np.array(targ_pred)
    target = targ_pred[:, 0]
    confidence = targ_pred[:, 1]
    # pre allocate
    num_pos = np.sum(target == 1)
    num_neg = np.sum(target == 0)
    TP, FP, last_TP = 0, 0, 0
    TPRs,FPRs = [],[]
    for i in range(1,len(target)):
        if (confidence[i] != confidence[i-1]) and (target[i] == 0) and (TP > last_TP):
            FPR = 1.0 * FP / num_neg
            TPR = 1.0 * TP / num_pos
            FPRs.append(FPR)
            TPRs.append(TPR)
            last_TP = TP
        if target[i] == 1:
            TP +=1
        else:
            FP +=1
    FPR = 1.0 * FP / num_neg
    TPR = 1.0 * TP / num_pos
    FPRs.append(FPR)
    TPRs.append(TPR)
    # plot ROC curve
    plt.figure(1)
    LW = 2.0
    plt.plot(TPRs, FPRs, marker='x', linewidth=LW)
    plt.title('%s' % dataName)
    plt.ylabel('True Positive Rate'); plt.xlabel('False Positive Rate')
    plt.show()


####----run----####

weight_lb = -1
weight_up = 1
rate = 0.1
if __name__ == '__main__':
    dataName, nFold, rate, nEpoch = loadDataName()
    THRESHOLD = 0.5
    data, metadata, feature_range = loadData(dataName)
    nHidden = len(metadata.types())
    labels = feature_range[-1]
    foldIndex = cvIndex(data, labels, nFold)
    predicts, confidences = [-1 for i in range(len(data))], [-1 for i in range(len(data))]
    for i in range(nFold):
        data_train, data_test = splitData(data, foldIndex, i)
        weights = trainModel(data_train, THRESHOLD, labels, metadata, nHidden, rate, nEpoch)
        predict, confidence = testModel(data_test, THRESHOLD, labels, metadata, weights, nHidden)
        index = 0
        for j in range(len(data)):
            if foldIndex[j] == i:
                predicts[j] = predict[index]
                index += 1
        index = 0
        for j in range(len(data)):
            if foldIndex[j] == i:
                confidences[j] = confidence[index]
                index += 1
    showresults(data, labels, metadata, foldIndex, predicts, confidences, show = True)


####----test----####

# nFold = 10
# rate = 0.1
# nEpoch = 100
# dataName = str("sonar.arff")
# weight_lb = -1
# weight_up = 1
# THRESHOLD = 0.5
# # nHidden = 10
# data, metadata, feature_range = loadData(dataName)
# nHidden = len(metadata.types())
# labels = feature_range[-1]
# foldIndex = cvIndex(data, nFold)
# predicts, confidences = [], []
# for i in range(nFold):
#     data_train, data_test = splitData(data, foldIndex, i)
#     weights = trainModel(data_train, nHidden, rate, nEpoch)
#     predict, confidence = testModel(data_test, weights, nHidden)
#     predicts.append(predict)
#     confidences.append(confidence)
# showresults(data, foldIndex, predicts, confidences)
    



# weights = trainModel(data_train,0,0.5,100)
# predict, confidence = testModel(data_test, weights, nHidden)
# X, Y = cleanData(data_test)

# predict, confidence = testModel(data_train, weights, nHidden)
# X, Y = cleanData(data_train)
# print(predict)
# print(confidence)





# _, TP, FP, FN, TN =showresults(data, labels, metadata, foldIndex, predicts, confidences, show = False)
#     if TP == 0 :
#         x = 0
#     else:
#         x = 1.*TP/(TP + FN)
#     if FP == 0:
#         y = 0
#     else:
#         y = 1.*FP/(FP + TN)
#     XX.append(x)
#     YY.append(y)

# YYs = sorted(YY)    
# order = [YYs.index(x) for x in YY]
# XXs = [XX[i] for i in order]


##----plot----


dataName = str("sonar.arff")
data, metadata, feature_range = loadData(dataName)
nHidden = len(metadata.types())
labels = feature_range[-1]
counts = []

predicts, confidences = [-1 for i in range(len(data))], [-1 for i in range(len(data))]
for nEpoch in [25, 50, 75, 100]:
    THRESHOLD = 0.5
    nFold = 10
    print(nEpoch)
    foldIndex = cvIndex(data, labels, nFold)
    for i in range(nFold):
        data_train, data_test = splitData(data, foldIndex, i)
        weights = trainModel(data_train, THRESHOLD, labels, metadata, nHidden, rate, nEpoch)
        predict, confidence = testModel(data_test, THRESHOLD, labels, metadata, weights, nHidden)
        index = 0
        for j in range(len(data)):
            if foldIndex[j] == i:
                predicts[j] = predict[index]
                index += 1
        index = 0
        for j in range(len(data)):
            if foldIndex[j] == i:
                confidences[j] = confidence[index]
                index += 1
    count = showresults(data, labels, metadata, foldIndex, predicts, confidences, show = False)
    count = 1.*count/len(data)
    counts.append(count)

plt.figure(1)
LW = 2.0
plt.plot([25, 50, 75, 100], counts, marker='x', linewidth=LW)
plt.title('%s' % dataName)
plt.ylabel('Accrucy'); plt.xlabel('Epoch')
plt.show()
from neuralnet import *



##----2----

predicts, confidences = [-1 for i in range(len(data))], [-1 for i in range(len(data))]
counts = []
for nFold in [5, 10, 15, 20, 25]:
    THRESHOLD = 0.5
    nEpoch = 50
    print(nFold)
    foldIndex = cvIndex(data, labels, nFold)
    for i in range(nFold):
        data_train, data_test = splitData(data, foldIndex, i)
        weights = trainModel(data_train, THRESHOLD, labels, metadata, nHidden, rate, nEpoch)
        predict, confidence = testModel(data_test, THRESHOLD, labels, metadata, weights, nHidden)
        index = 0
        for j in range(len(data)):
            if foldIndex[j] == i:
                predicts[j] = predict[index]
                index += 1
        index = 0
        for j in range(len(data)):
            if foldIndex[j] == i:
                confidences[j] = confidence[index]
                index += 1
    count = showresults(data, labels, metadata, foldIndex, predicts, confidences, show = False)
    count = 1.*count/len(data)
    counts.append(count)

plt.figure(2)
LW = 2.0
plt.plot([5, 10, 15, 20, 25], counts, marker='x', linewidth=LW)
plt.title('%s' % dataName)
plt.ylabel('Accrucy'); plt.xlabel('Fold')
plt.show()



##----3----

predicts, confidences = [-1 for i in range(len(data))], [-1 for i in range(len(data))]
XX = []
YY = []
THRESHOLD = 0.5
nEpoch = 50
nFold = 10
foldIndex = cvIndex(data, labels, nFold)
for i in range(nFold):
    data_train, data_test = splitData(data, foldIndex, i)
    weights = trainModel(data_train, THRESHOLD, labels, metadata, nHidden, rate, nEpoch)
    predict, confidence = testModel(data_test, THRESHOLD, labels, metadata, weights, nHidden)
    index = 0
    for j in range(len(data)):
        if foldIndex[j] == i:
            predicts[j] = predict[index]
            index += 1
    index = 0
    for j in range(len(data)):
        if foldIndex[j] == i:
            confidences[j] = confidence[index]
            index += 1

_, Y = cleanData(data, labels, metadata)
plotROC(Y, predicts, confidences)