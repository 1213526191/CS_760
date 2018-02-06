import matplotlib.pyplot as plt
import numpy as np
from neuralnet import *

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