
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import feature_extraction



#: Some helper functions

#: modified log2 function to account for when proportion == 0
def log2_modified(proportion):
    if proportion == 0:
        return 0
    else:
        return np.log2(proportion)


class Node(object):
    def __init__(self, split_rule, left, right, label=None, purity=-1):
        self.split_rule = split_rule
        self.left = left
        self.right = right
        self.label = label
        self.purity = purity

        # self.depth = depth



class DecisionTree(object):

    def __init__(self, isBinaryFeature, useRandomSubsetOfFeatures=False, maxDepth=10):
        """

        :param isBinaryFeature: list of length numFeatures; indices: featureIndex ; value: True if i'th feature is a binary feature
        """
        self.rootNode = None
        self.isBinaryFeature = isBinaryFeature
        self.useRandomSubsetOfFeatures = useRandomSubsetOfFeatures #for random forest functionality

        self.maxDepth = maxDepth

    @staticmethod
    def pureSet(labels):
        """
        Checks if the set S is pure (all elements here all have the same label). Assumes labels are either 0 or 1
        :param node:
        :return: isPure, label, purity
        """
        labels_sum = labels.sum()
        numLabels = labels.shape[0]

        if labels_sum == numLabels:
            return True, 1, 1.0
        elif labels_sum == 0:
            return True, 0, 1.0
        else:
            if labels_sum > 0.5 * numLabels:
                label_majority = 1
                purity = 1.0 * labels_sum / numLabels
            else:
                label_majority = 0
                purity = 1.0 - (1.0 * labels_sum / numLabels)
            return False, label_majority, purity

    @staticmethod
    def split(split_rule, data, labels):
        """
        Splits the data based on the split rule, into left and right datasets
        :param split_rule:
        :param data:
        :param labels:
        :return: left_data, left_labels, right_data, right_labels
        """
        splitFeature, splitThreshold = split_rule

        left_data = np.zeros((0, data.shape[1]))
        left_labels_lst = []
        right_data = np.zeros((0, data.shape[1]))
        right_labels_lst = []

        for i in np.arange(data.shape[0]):
            if data[i, splitFeature] < splitThreshold:
                left_data = np.vstack((left_data, data[i, :]))
                left_labels_lst += [labels[i]]
            else:
                right_data = np.vstack((right_data, data[i, :]))
                right_labels_lst += [labels[i]]

        return left_data, np.array(left_labels_lst), right_data, np.array(right_labels_lst)


    @staticmethod
    def impurity(left_label_hist, right_label_hist):
        """
        Calculates the impurity (badness) of a split. Uses entropy measure.
        :param left_label_hist: an array where key is 0 or 1, value is count
        :param right_label_hist:
        :return: impurity
        """

        total_count = left_label_hist[0] + right_label_hist[0] + left_label_hist[1] + right_label_hist[1]
        # p_0_S = 1.0 * (left_label_hist[0] + right_label_hist[0]) / total_count #p_c is proportion of points in S that are in class C
        # p_1_S = 1.0 * (left_label_hist[1] + right_label_hist[1]) / total_count
        # H_S = -(p_0_S * log2_modified(p_0_S) + p_1_S * log2_modified(p_1_S)) # H(S), i.e. the entropy of S

        left_hist_count = (left_label_hist[0] + left_label_hist[1])
        if left_hist_count > 0:
            p_0_S_left = 1.0 * left_label_hist[0] / left_hist_count
            p_1_S_left = 1.0 * left_label_hist[1] / left_hist_count
            H_S_left = -(p_0_S_left * log2_modified(p_0_S_left) + p_1_S_left * log2_modified(p_1_S_left))  # H(S_l), i.e. the entropy of S_l
        else:
            H_S_left = 0

        right_hist_count = (right_label_hist[0] + right_label_hist[1])
        if right_hist_count > 0:
            p_0_S_right = 1.0 * right_label_hist[0] / right_hist_count
            p_1_S_right = 1.0 * right_label_hist[1] / right_hist_count
            H_S_right = -(p_0_S_right * log2_modified(p_0_S_right) + p_1_S_right * log2_modified(p_1_S_right))  # H(S_l), i.e. the entropy of S_l
        else:
            H_S_right = 0

        H_after = (left_hist_count * H_S_left + right_hist_count * H_S_right) / total_count

        # info_gain = H_S - H_after
        # return -(info_gain)

        return H_after

    def segmenter(self, data, labels):
        numFeatures = data.shape[1]

        bestImpurity = float('inf')
        bestFeature = None # index of best feature
        bestThreshold = None # the elements with the bestThreshold value are in the right-hand set

        if not self.useRandomSubsetOfFeatures:
            features = np.arange(numFeatures)
        else:
            features = np.random.choice(np.arange(numFeatures), int(np.ceil(numFeatures**0.5)), replace=False)

        for i in features:
            left_label_hist = np.array([0, 0])
            right_label_hist = np.array([labels.shape[0] - labels.sum(), labels.sum()])

            if not self.isBinaryFeature[i]: # continuous feature
                sorted_vals = np.sort(data[:, i])
                indices = np.argsort(data[:, i])

                j = 0
                currentStartVal = None # keep track of this through the 'while' loop so we can handle repeated vals
                numSamples = sorted_vals.shape[0]
                while j < numSamples:

                    #: try to add all repeated elements into left hist
                    currentStartVal = sorted_vals[j-1]
                    currentVal = currentStartVal
                    while currentVal == currentStartVal:
                        if j != 0:
                            label_j1 = labels[indices[j-1]] # label of (j-1)th elem in the sorted_vals array
                            left_label_hist[label_j1] += 1
                            right_label_hist[label_j1] -= 1
                        if j < numSamples:
                            currentVal = sorted_vals[j]
                            j += 1
                        else:
                            break

                    imp = DecisionTree.impurity(left_label_hist, right_label_hist)
                    if imp < bestImpurity:
                        bestImpurity = imp
                        bestFeature = i
                        if j < numSamples:
                            bestThreshold = sorted_vals[j]
                        else:
                            bestThreshold = sorted_vals[j-1] + 1
            else:  # binary feature
                for threshold in [0, 1]:
                    if threshold == 1:
                        # add all elems with feature as value 0 (NOT LABEL) to left hist
                        for k in np.arange(data[:, i].shape[0]):
                            if data[k, i] == 0:
                                left_label_hist[labels[k]] += 1
                                right_label_hist[labels[k]] -= 1
                    imp = DecisionTree.impurity(left_label_hist, right_label_hist)
                    if imp < bestImpurity:
                        bestImpurity = imp
                        bestFeature = i
                        bestThreshold = threshold

        if bestFeature == None or bestThreshold == None:
            print 'Error: in segmenter(.), bestFeature or bestThreshold were not found.'

        return bestFeature, bestThreshold


    def train(self, data, labels):
        self.rootNode = self.growTree(data, labels, depth=0)


    def growTree(self, data, labels, depth):
        # Based on the GrowTree method described in lecture notes
        isPure, label, purity = DecisionTree.pureSet(labels)
        if purity > 0.80 or depth > self.maxDepth: #if isPure or self.numNodeSplits > self.maxNumNodeSplits
            return Node(split_rule=None, left=None, right=None, label=label, purity=purity)
        else:
            split_rule = self.segmenter(data, labels)
            left_data, left_labels, right_data, right_labels = DecisionTree.split(split_rule, data, labels)
            # print 'depth: ', depth, '; purity: ', purity, '; #leftlabels: ', left_labels.shape[0], '; #rightlabels: ', right_labels.shape[0] #@@@

            return Node(split_rule, self.growTree(left_data, left_labels, depth + 1), self.growTree(right_data, right_labels, depth + 1), purity=purity)


    def predict(self, data):
        predicted_labels = np.zeros((data.shape[0]))
        for i in np.arange(data.shape[0]):
            node = self.rootNode
            while node.label is None:
                splitFeature, splitThreshold = node.split_rule
                if data[i, splitFeature] < splitThreshold:
                    node = node.left
                else:
                    node = node.right
            predicted_labels[i] = node.label

        return predicted_labels


class RandomForest(object):
    def __init__(self, isBinaryFeature, numTrees=10, maxDepth=10):
        self.numTrees = numTrees
        self.isBinaryFeature = isBinaryFeature
        self.trees = []
        self.maxDepth = maxDepth

    def train(self, data, labels):
        for i in np.arange(self.numTrees):
            print 'Creating tree ', i

            tree = DecisionTree(self.isBinaryFeature, useRandomSubsetOfFeatures=True, maxDepth=self.maxDepth)

            numSamples = data.shape[0]
            numFeatures = data.shape[1]

            #: create random subset of the data
            sampledIndices = np.random.choice(np.arange(numSamples), numSamples, replace=True) #here, n == n'
            data_new = np.zeros((0, numFeatures))
            labels_new_lst = []

            for j in sampledIndices:
                data_new = np.vstack((data_new, data[j, :]))
                labels_new_lst += [labels[j]]
            labels_new = np.array(labels_new_lst)

            tree.train(data_new, labels_new)

            self.trees += [tree]

    def predict(self, data):
        numSamples = data.shape[0]

        predictedLabels_allTrees = np.zeros((self.numTrees, numSamples))

        for i in np.arange(len(self.trees)):
            predictedLabels_allTrees[i, :] = self.trees[i].predict(data)

        predictedLabels = np.round(predictedLabels_allTrees.mean(0))

        return predictedLabels
