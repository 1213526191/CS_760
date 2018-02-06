import sys
import numpy as np
import scipy.io.arff as sparff

def loadDataName():
    data_train = str(sys.argv[1])
    data_test = str(sys.argv[2])
    option = str(sys.argv[3])
    return data_train, data_test, option

def loadData(dataName):
    data, metadata = sparff.loadarff(dataName)
    nFeature = len(data[0]) - 1
    nInstance = len(data)
    X, Y = [], []
    for i in range(nInstance):
        Y.append(metadata[metadata.names()[-1]][1].index(data[i][-1]))
        variables = []
        for j in range(nFeature):
            variables.append(metadata[metadata.names()[j]][1].index(data[i][j]))
        X.append(variables)
    feature_range = []
    for i in range(nFeature + 1):
        feature_range.append(len(metadata[metadata.names()[i]][1]))
    return np.array(X), np.array(Y), metadata, feature_range

def getDist(Y, feature_range_Y):
    counts = np.zeros(feature_range_Y, )
    for j in Y:
        counts[j] += 1
    counts += 1
    dist = counts/sum(counts)
    return dist

def getSubset(X, Y, y):
    index = Y == y
    X_sub = X[index]
    return X_sub

def computeP_XgY(X, Y, feature_range):
    nFeature = len(feature_range) - 1
    levels = feature_range[-1]
    P_XgY = [[] for i in range(levels)]
    for y in range(levels):
        X_sub = getSubset(X, Y, y)
        for i in range(nFeature):
            X_sub_sub = [X_sub[j][i] for j in range(len(X_sub))]
            dist = getDist(X_sub_sub, feature_range[i])
            P_XgY[y].append(dist)
    return P_XgY
        
def bayesDist(X, Y, feature_range):
    P_Y = getDist(Y, feature_range[-1])
    P_XgY = computeP_XgY(X, Y, feature_range)
    return P_Y, P_XgY




def navieBayesPrediction(X_test, P_Y, P_XgY, feature_range):
    Y_pred = []
    Y_prob = []
    nInstance = len(X_test)
    nFeature = len(X_test[0])
    for i in range(nInstance):
        prob = [0 for k in range(feature_range[-1])]
        for y in range(feature_range[-1]):
            prob[y] = P_Y[y]
            for index, value in enumerate(X_test[i]):
                prob[y] *= P_XgY[y][index][value]
        dist = np.divide(prob, np.sum(prob))
        Y_hat = np.argmax(prob)
        Y_prob_ = dist[Y_hat]
        Y_pred.append(Y_hat)
        Y_prob.append(Y_prob_)
    return Y_pred, Y_prob
            
def print1(metadata):
    for i in range(len(metadata.names()) - 1):
        feature_info = metadata[metadata.names()[i]]
        print('%s %s' % (metadata.names()[i], metadata.names()[-1]))       
    return 0
    
def printTestResults(Y_hat, Y_prob, Y_test, metadata):
    y_range = metadata[metadata.names()[-1]][1]
    for m in range(len(Y_test)):
        prediction = y_range[int(Y_hat[m])]
        truth = y_range[Y_test[m]]
        print('%s %s %.12f' % (prediction.strip('"\''), truth.strip('"\''), Y_prob[m]))
    hits = np.sum(np.around(Y_hat) == Y_test)
    print ('\n%d' % hits)


## ----TAN----

def computeP_XiXjgYk(Xi, Xj, feature_range_i, feature_range_j):
    poss = np.zeros((feature_range_i, feature_range_j))
    for xi in range(feature_range_i):
        for xj in range(feature_range_j):
            for m in range(len(Xi)):
                if Xi[m] == xi and Xj[m] == xj:
                    poss[xi][xj] += 1
    poss += 1
    dist = np.divide(poss, sum(sum(poss)))
    return dist
    

def computeP_XXgY(X, Y, feature_range):
    nFeature = len(X[0])
    counts = [[] for k in range(feature_range[-1])]
    for k in range(feature_range[-1]):
        count = [[[] for i in range(nFeature)] for j in range(nFeature)]
        for i in range(nFeature):
            for j in range(nFeature):
                index = Y == k
                X_ = X[index]
                count[i][j] = computeP_XiXjgYk(X_[:,i], X_[:,j], feature_range[i], feature_range[j])
        counts[k] = count
    return counts

def computeP_XiXjYk(Xi, Xj, feature_range_i, feature_range_j, Y, feature_range_y):
    count = np.zeros((feature_range_i, feature_range_j, feature_range_y))
    # count = [[[0 for y in range(feature_range_y)] for j in range(feature_range_j)] for i in range(feature_range_i)]
    for i in range(feature_range_i):
        for j in range(feature_range_j):
            for y in range(feature_range_y):
                for m in range(len(Xi)):
                    if Xi[m] == i and Xj[m] == j and Y[m] == y:
                        count[i][j][y] += 1
    count += 1
    dist = np.divide(count, sum(sum(sum(count))))
    return dist

def computeP_XXY(X, Y, feature_range):
    nFeature = len(X[0])
    counts = [[[] for i in range(nFeature)] for j in range(nFeature)]
    for i in range(nFeature):
        for j in range(nFeature):
            count = computeP_XiXjYk(X[:,i], X[:,j], feature_range[i], feature_range[j], Y, feature_range[-1])
            counts[i][j] = count
    return counts

def computeMIij(P_XgY_i, P_XgY_j, P_XXgY_ij, P_XXY_ij, feature_range_i, feature_range_j, y):
    MIij = 0
    for xi in range(feature_range_i):
        for xj in range(feature_range_j):
            # print str(xi)+" "+str(xj)
            MIij += P_XXY_ij[xi][xj][y]*np.log2(P_XXgY_ij[xi][xj]/P_XgY_i[xi]/P_XgY_j[xj])
    return MIij

def computeMI(P_XgY, P_XXgY, P_XXY, feature_range):
    n = len(feature_range) - 1
    MI = np.zeros((n,n))
    for i in range(n):
        for j in range((i + 1), n):
            for y in range(feature_range[-1]):
                # print str(i)+" "+str(j)
                MI[i][j] += computeMIij(P_XgY[y][i], P_XgY[y][j], P_XXgY[y][i][j], P_XXY[i][j], feature_range[i], feature_range[j], y)
    for i in range(n):
        MI[i, i] = -1
        for j in range(i):
            MI[i][j] = MI[j][i]
    return MI

def tanStructure(X, Y, metadata, feature_range):
    P_XgY = computeP_XgY(X, Y, feature_range)
    P_XXgY = computeP_XXgY(X, Y, feature_range)
    P_XXY = computeP_XXY(X, Y, feature_range)
    MI = computeMI(P_XgY, P_XXgY, P_XXY, feature_range)
    parent = computeParent(MI)
    return parent

def computeParent(MI): 
    vex_num = len(MI)  
    prims = [0 for i in range(vex_num)]  
    weights = [0 for i in range(vex_num)]  
    flag_list = [False]*vex_num  
    node = 0  
    flag_list[node] = True  
    for i in range(vex_num):  
        weights[i] = MI[node][i]      
    for i in range(vex_num - 1):  
        max_value = 0  
        for j in range(vex_num):  
            if weights[j] > max_value and not flag_list[j]:  
                max_value = weights[j]  
                node = j  
        if node == 0:  
            return  
        flag_list[node] = True  
        for m in range(vex_num):  
            if weights[m] < MI[node][m] and not flag_list[m]:  
                weights[m] = MI[node][m]  
                prims[m] = node   
    return prims  

def printGraph_TAN(parent, metadata):
    featureNames = metadata.names()
    for n in range(len(featureNames) -1):
        # print the immediate parent, follow by Y
        if parent[n] is n:
            print ('%s %s' % (featureNames[n], featureNames[-1]))
        else:
            print ('%s %s %s'% (featureNames[n], featureNames[parent[n]], featureNames[-1]))
    print

def computeP_XigXjY(Xi, Xj, feature_range_i, feature_range_j, Y, feature_range_y):  
    P_XigXjY = [[[] for i in range(feature_range_y)] for j in range(feature_range_j)]
    for xj in range(feature_range_j):
        for y in range(feature_range_y):
            index1 = Y == y
            index2 = Xj == xj
            Xi_sub = Xi[index1 * index2]
            count = np.zeros(feature_range_i)
            for xi in range(feature_range_i):
                for m in Xi_sub:
                    if m == xi:
                        count[xi] += 1
            count += 1
            dist = np.divide(count, sum(count))
            P_XigXjY[xj][y] = dist
    return P_XigXjY



# def computeP_XgXY(X, Y, feature_range, i, parent):
#     n = len(X[0])
#     P_XgXY = [[[] for i in range(n)] for j in range(n)]
#     for i in range(n):
#         for j in range(n):
#             P_XgXY[i][j] = computeP_XigXjY(X[:, i], X[:, j], feature_range[i], feature_range[j], Y, feature_range[-1])
#     return P_XgXY

def computeP_XigY(Xi, Y, feature_range_i, feature_range_y):
    P_XigY = [[] for y in range(feature_range_y)]
    for y in range(feature_range_y):
        index = Y == y
        Xi_sub = Xi[index]
        count = np.zeros(feature_range_i)
        for xi in range(feature_range_i):
            for m in Xi_sub:
                if m ==xi:
                    count[xi] += 1
        count += 1
        dist = np.divide(count, sum(count))
        P_XigY[y] = dist
    return P_XigY

def computeP_XgXY(X, Y, feature_range, parent):
    n = len(parent)
    P_XgXY = [[] for i in range(n)]
    for i in range(n):
        if parent[i] is i:
            Xi = X[:, i]
            feature_range_i = feature_range[i]
            feature_range_y = feature_range[-1]
            P_XgXY[i] = computeP_XigY(Xi, Y, feature_range_i, feature_range_y)
        else:
            j = parent[i]
            Xi = X[:, i]
            Xj = X[:, j]
            feature_range_i = feature_range[i]
            feature_range_j = feature_range[j]
            feature_range_y = feature_range[-1]
            P_XgXY[i] = computeP_XigXjY(Xi, Xj, feature_range_i, feature_range_j, Y, feature_range_y)
    P_XgXY.append(getDist(Y, feature_range[-1]))
    return P_XgXY
    
def computePredictions_TAN(X_test, P_XgXY, parent, feature_range):
    Y_hat = np.zeros(len(X_test))
    Y_prob = np.zeros(len(X_test))
    for i, x in enumerate(X_test):
        # print i
        p = np.zeros(feature_range[-1])
        for k in range(feature_range[-1]):
            p[k] = P_XgXY[-1][k]
        for index, value in enumerate(x):
            # print p
            # print str(index)+" "+str(value)
            for y in range(feature_range[-1]):
                if parent[index] is index:             
                    p[y] *= P_XgXY[index][y][value]
                else:
                    j = parent[index]
                    p[y] *= P_XgXY[index][x[j]][y][value]
        dist = np.divide(p, sum(p))
        Y_prob[i] = max(dist)
        Y_hat[i] = np.argmax(dist)
    return Y_hat, Y_prob
            


