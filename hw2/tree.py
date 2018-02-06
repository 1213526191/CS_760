import numpy as np

def isNumeric(type_str):
    if type_str == "numeric" or type_str == "real" or type_str == "integer":
        return True
    else:
        return False

class decisionTreeNode:
    def __init__(self, i = 0, feature_name = None, feature_value = None, threshold = None, parent = None, 
    children = None, classification = None, feature_used = None, isLeaf = None, isRoot = None):

        self.feature_name = feature_name
        self.threshold = threshold
        self.parent = parent
        self.children = []
        self.classification = classification
        self.feature_used = feature_used
        self.isLeaf = isLeaf
        self.isRoot = False
        self.counts = np.array(2,)
        self.depth = 0
        self.isLeftChild = False
        self.i = i

    def getFeatureName(self):
        return self.feature_name    
    def getFeatureValue(self):
        return self.feature_value
    def getThreshold(self):
        return self.threshold
    def getParent(self):
        return self.parent
    def getChildren(self):
        return self.children
    def getClassification(self):
        return self.classification
    def getFeatureUsed(self):
        return self.feature_used
    def isTerminalNode(self):
        if self.isLeaf:
            return True
        return False
    def isRoot(self):
        if self.isRoot:
            return True
        return False
    def getCounts(self):
        return self.counts
    def isLeftChild(self):
        return self.isLeftChild

    def setRoot(self, isRoot_):
        self.isRoot = isRoot_
    def setLeftChild(self):
        self.isLeftChild = True
    def setFeature(self, _feature_name, _threshold):
        self.feature_name = _feature_name
        self.threshold = _threshold
    def setThreshold(self, threshold_):
        self.threshold = threshold_
    def updateUsedFeature(self, feature_used_):
        self.feature_used = feature_used_
    def setChildren(self, _children):
        self.children.append(_children)
    def setParent(self, _parent):
        self.parent = _parent
    def setClassificationLabel(self, _classLabel):
        self.classification = _classLabel
    def setLeaf(self):
        self.isLeaf = True
    def setDepth(self, depth_):
        self.depth = depth_
    def setCounts(self, counts):
        self.counts = counts
