import numpy as np
import scipy.io.arff as sparff
import scipy as sp
import sys
import matplotlib.pyplot as plt

from tree import decisionTreeNode

def isNumeric(type_str):
    if type_str == "numeric" or type_str == "real" or type_str == "integer":
        return True
    else:
        return False

def loadDataName():
    if len(sys.argv) == 4:
        dataName_train = str(sys.argv[1])
        dataName_test = str(sys.argv[2])
        m = int(str(sys.argv[3]))
    else:
        sys.exit('ERROR: This program takes exactly 3 input arguments.')     
    return dataName_train, dataName_test, m

def loadData(data_):
    data, metadata = sparff.loadarff(data_)
    # change data to UTF-8
    col_name_t = [name for name in metadata.names()]
    data_category_t = [metadata[name][1] for name in col_name_t]
    str_index_t = [index for index, content in enumerate(metadata.types()) if content == 'nominal']
    data_class = data_category_t[-1]
    data_raw_t = data.tolist()
    for i in range(len(data_raw_t)):
        data_raw_t[i] = list(data_raw_t[i])
        for j in str_index_t:
            data_raw_t[i][j] = data_raw_t[i][j].decode()
    data = np.array(data_raw_t, dtype = 'O')
    
    feature_range = []
    for name in metadata.names():
        feature_range.append(metadata[name][1])
    return data, metadata, feature_range

def informationGain(data_, feature_used, feature_range, metadata):
    entropyX = getEntropyX(data_, feature_used, feature_range, metadata)
    entropyY = getEntropyY(data_)
    informationGain_ = np.subtract(entropyY, entropyX)
    return informationGain_

def getEntropyY(data_):
    ylabels = []
    for instance in data_:
        ylabels.append(instance[-1])
    counts = 0
    for label in ylabels:
        if label == labelRange[0]:
            counts += 1
    if counts == 0 or counts == len(ylabels):
        return 0
    else:
        p = counts*1.0/len(ylabels)
        return -p * np.log2(p) - (1-p) * np.log2(1-p)

def getEntropyX(data_, feature_used_, feature_range, metadata):
    n = len(metadata.types()) - 1
    entropy = np.zeros(n)
    entropy.fill(np.NaN)
    for i in range(n):
        if feature_used_[i]:
            continue
        elif isNumeric(metadata.types()[i]):
            threshold, entropy[i] = threshold_C(data_, i)
        else:
            entropy[i] = threshold_D(data_, i, feature_range[i])
    return entropy

def threshold_C(data_, i):
    feature_values = []
    for instance in data_:
        feature_values.append(instance[i])
    feature_values = np.array(feature_values)
    uniqueFeature_values = np.sort(np.unique(feature_values))
    splitPoints = np.divide(np.add(uniqueFeature_values[0:-1], uniqueFeature_values[1:]), 2.)
    bestThreshold = 0
    minEntropy = float('inf')
    for threshold in splitPoints:
        data1, data2 = splitData_C(data_, i, threshold)
        entropy = (getEntropyY(data1)*len(data1) + getEntropyY(data2)*len(data2))/len(data_)
        if entropy < minEntropy:
            minEntropy = entropy
            bestThreshold = threshold
    return bestThreshold, minEntropy

def splitData_C(data_, i, threshold):
    data1 = []
    data2 = []
    for instance in data_:
        if(instance[i] <= threshold):
            data1.append(instance)
        else:
            data2.append(instance)
    return data1, data2

def threshold_D(data_, i, feature_range):
    entropy = 0
    for label in feature_range:
        data1 = []
        for instance in data_:
            if instance[i] == label:
                data1.append(instance)
        if not len(data1) == 0:
            entropy += getEntropyY(data1)*len(data1)/len(data_)
    return entropy

def splitData_D(data_, i):
    data_divided = []
    for label in feature_range[i]:
        data1 = []
        for instance in data_:
            if instance[i] == label:
                data1.append(instance)
        data_divided.append(data1)
    return data_divided

def printTree(node, depth = 0):
    if not node.isRoot:
        [count1, count2] = node.getCounts()
        x = metadata[node.getFeatureName()][0]
        if isNumeric(x):
            if node.isLeftChild:
                equality = "<="
            else:
                equality = ">"
            threshold = "%.6f" % node.getThreshold() # Six decimal
        else:
            equality = "="
            threshold = node.getThreshold()
        if node.isTerminalNode():
            print(depth * "|\t" + "%s %s %s [%d %d]: %s" \
                                  % (node.getFeatureName(), equality, threshold,
                                     count1, count2, node.getClassification()))
        else:
            print(depth * "|\t" + "%s %s %s [%d %d]" \
                                  % (node.getFeatureName(), equality, threshold,
                                     count1, count2))
        depth +=1
    for child in node.getChildren():
        printTree(child, depth)
            
def makeTree(data_, feature_name, threshold, feature_used, parent_ = None,
                            isRoot_ = True, isLeftChild_ = False):
    informationGain_ = informationGain(data_, feature_used, feature_range, metadata)
    if stopGrow(informationGain_, data_, feature_used):
        isLeaf_ = True
        return nodeInfo(data_, feature_name, threshold, feature_used, parent_,
                            isRoot_, isLeaf_, isLeftChild_)
    else:
        isLeaf_ = False
        node = nodeInfo(data_, feature_name, threshold, feature_used, parent_,
                            isRoot_ , isLeaf_ , isLeftChild_)
        informationGain_2 = informationGain_[~np.isnan(informationGain_)]
        a = list(informationGain_)
        b = list(informationGain_2)
        best_feature_index = a.index(max(b))
        feature_name = metadata.names()[best_feature_index]
        # feature_used[best_feature_index] = True
        if isNumeric(metadata.types()[best_feature_index]):
            threshold, _ = threshold_C(data_, best_feature_index)
            data_left, data_right = splitData_C(data_, best_feature_index, threshold)
            if not len(data_left) == 0:
                feature_used[best_feature_index] = True
                child_left = makeTree(data_left, feature_name, threshold, 
                feature_used, parent_ = node, isRoot_ = False, isLeftChild_ = True)
                feature_used[best_feature_index] = False
                node.setChildren(child_left)
            if not len(data_right) == 0:
                feature_used[best_feature_index] = True
                child_right = makeTree(data_right, feature_name, threshold, 
                feature_used, parent_ = node, isRoot_ = False, isLeftChild_ = False)
                feature_used[best_feature_index] = False
                node.setChildren(child_right)
        else:
            data_divided = splitData_D(data_, best_feature_index)
            n = len(feature_range[best_feature_index])          
            for i in range(n):
                threshold = feature_range[best_feature_index][i]
                data1 = data_divided[i]
                feature_used[best_feature_index] = True
                child = makeTree(data1, feature_name, 
                threshold, feature_used, node, isRoot_ = False)
                feature_used[best_feature_index] = False
                node.setChildren(child)
    return node
      
def nodeInfo(data_, feature_name, threshold, feature_used, parent_,
                            isRoot_ , isLeaf_ , isLeftChild_ = False):
    node = decisionTreeNode()
    if isRoot_:
        count1, count2 = getCounts(data_)
        if count1 >= count2:
            label = labelRange[0]
        else:
            label = labelRange[1]
        counts = [count1, count2]
    else:
        label, counts = getMajority(data_, parent_)
    node.setFeature(feature_name, threshold)
    node.setParent(parent_)
    node.setClassificationLabel(label)
    node.setCounts(counts)
    node.setRoot(isRoot_)
    if isLeaf_:
        node.setLeaf()
    if isLeftChild_:
        node.setLeftChild()
    return node



def getCounts(data_):
    count1 = 0
    count2 = 0
    for instance in data_:
        if instance[-1] == labelRange[0]:
            count1 += 1
        else:
            count2 += 1
    return count1, count2

def getMajority(data_, parent_):
    count1, count2 = getCounts(data_)
    if count1 > count2:
        majorityClassLabel = labelRange[0]
    elif count1 < count2:
        majorityClassLabel = labelRange[1]
    else:
        majorityClassLabel = parent_.getClassification()
    return majorityClassLabel, [count1, count2]

def stopGrow(informationGain_, data_, feature_used_):
    ylabels = []
    for instance in data_:
        ylabels.append(instance[-1]) 
    informationGain_ = np.array(informationGain_)
    informationGain_ = informationGain_[~np.isnan(informationGain_)]
    if len(set(ylabels)) == 1 or len(data_)< m or all(sp.less(informationGain_, 0)) or all(feature_used_):
        return True
    return False

def classify(instance, node):
    prediction = None
    if node.isTerminalNode():
        return node.classification
    for child in node.children:
        feature_name = child.feature_name
        feature_type = metadata[feature_name][0]
        feature_intex = all_featureNames.index(feature_name)
        if isNumeric(feature_type):
            if(child.isLeftChild):
                if instance[feature_intex] <= child.getThreshold():
                    prediction = classify(instance, child)
            else:
                if instance[feature_intex] > child.getThreshold():
                    prediction = classify(instance, child)
        else:
            if instance[feature_intex] == child.getThreshold():
                prediction = classify(instance, child)
    return prediction

def printTestPerformance(data_test, decisionTree, printResults = False):
    if printResults:
        print("<Predictions for the Test Set Instances>")
    correctCount = 0
    numData = len(data_test)
    for i in range(numData):
        instance = data_test[i]
        y_pred = classify(instance, decisionTree)
        y_actual = instance[-1]
        if y_pred == y_actual:
            correctCount +=1
        if printResults:
            print("%d: Actual: %s Predicted: %s" % (i+1, y_actual, y_pred))
    if printResults:
        print("Number of correctly classified: %d Total number of test instances: %d" \
          % (correctCount, numData))
    classification_accuracy = 1.0 * correctCount / numData
    return classification_accuracy
                    



##############


# visualizeResults = 1
# dataName_train = str("heart_train.arff")#
# m = 20
dataName_train, dataName_test, m = loadDataName()
data_train, metadata, feature_range = loadData(dataName_train)
# data_ = data_train#
feature_used = np.zeros((len(metadata.types()) - 1,), dtype=bool)
labelRange = feature_range[-1]
all_featureNames = list(metadata.names())

threshold = None
feature_name = None

decisionTree = makeTree(data_train, feature_name, threshold, feature_used)
printTree(decisionTree)

data_test, _, _= loadData(dataName_test)
printTestPerformance(data_test, decisionTree, True)



##########################



def randomSample(data, prop):
    n = len(data)
    m = int(round(prop*n))
    subset_index = np.random.choice(n, m, replace = False)
    subset = []
    for i in subset_index:
        subset.append(data[i])
    subset = np.array(subset)
    return(subset)


m = 4
sampleSize = 10
proportion = np.array([.05, .1, .2, .5, 1])
k = len(proportion)
accuracy = np.zeros((sampleSize, k))
for i in range(sampleSize):
    for j in range(k):
        subset = randomSample(data_train, proportion[j])
        feature_used = np.zeros((len(metadata.types()) - 1,), dtype=bool)
        feature_val_cur = None
        feature_name = None
        tree = makeTree(subset, feature_name, threshold, feature_used)
        accuracy[i,j] = printTestPerformance(data_test, tree)
        del tree

plt.figure(1)
plt.plot(range(k), np.mean(accuracy, 0))
plt.plot(range(k), np.amax(accuracy,0))
plt.plot(range(k), np.amin(accuracy,0))
plt.xticks(range(k), proportion)
plt.title('%s' % dataName_test)
plt.ylabel('Test set classification accuracy')
plt.xlabel('Proportion of training data')
plt.show()


# ################

mValue = np.array([2,5,10,20])
length = len(mValue)
accuracy = np.zeros(length)
for i in range(length):
    m = mValue[i]
    feature_used = np.zeros((len(metadata.types()) - 1,), dtype=bool)
    feature_val_cur = None
    feature_name = None
    tree = makeTree(data_train, feature_name, threshold, feature_used)
    accuracy[i] = printTestPerformance(data_test, tree)

plt.figure(2)
plt.plot(range(length), accuracy)
plt.xticks(range(length), mValue)
plt.title('%s' % dataName_test)
plt.ylabel('Test set classification accuracy')
plt.xlabel('m')
plt.show()
