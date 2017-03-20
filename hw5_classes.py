
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
    def __init__(self, split_rule, left, right, label=None):
        self.split_rule = split_rule
        self.left = left
        self.right = right
        self.label = label

class DecisionTree(object):

    def __init__(self, isBinaryFeature=None):
        """

        :param isBinaryFeature: list of length numFeatures; indices: featureIndex ; value: True if i'th feature is a binary feature
        """
        self.rootNode = None
        if isBinaryFeature is None:
            self.isBinaryFeature = []
        else:
            self.isBinaryFeature = isBinaryFeature

    @staticmethod
    def pureSet(labels):
        """
        Checks if the set S is pure (all elements here all have the same label). Assumes labels are either 0 or 1
        :param node:
        :return: isPure, label
        """
        labels_sum = labels.sum()
        if labels_sum == labels.shape[0]:
            return True, 1
        elif labels_sum == 0:
            return True, 0
        else:
            return False, -1

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

        left_data = np.array((0, data.shape[1]))
        left_labels_lst = []
        right_data = np.array((0, data.shape[1]))
        right_labels_lst = []

        for i in data.shape[0]:
            if data[i, splitFeature] < splitThreshold:
                np.vstack((left_data, data[i, :]))
                left_labels_lst += [labels[i]]
            else:
                np.vstack((right_data, data[i, :]))
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
        p_0_S = 1.0 * (left_label_hist[0] + right_label_hist[0]) / total_count #p_c is proportion of points in S that are in class C
        p_1_S = 1.0 * (left_label_hist[1] + right_label_hist[1]) / total_count
        H_S = -(p_0_S * log2_modified(p_0_S) + p_1_S * log2_modified(p_1_S)) # H(S), i.e. the entropy of S

        left_hist_count = (left_label_hist[0] + left_label_hist[1])
        p_0_S_left = 1.0 * left_label_hist[0] / left_hist_count
        p_1_S_left = 1.0 * left_label_hist[1] / left_hist_count
        H_S_left = -(p_0_S_left * log2_modified(p_0_S_left) + p_1_S_left * log2_modified(p_1_S_left))  # H(S_l), i.e. the entropy of S_l

        right_hist_count = (right_label_hist[0] + right_label_hist[1])
        p_0_S_right = 1.0 * right_label_hist[0] / right_hist_count
        p_1_S_right = 1.0 * right_label_hist[1] / right_hist_count
        H_S_right = -(p_0_S_right * log2_modified(p_0_S_right) + p_1_S_right * log2_modified(p_1_S_right))  # H(S_l), i.e. the entropy of S_l

        H_after = (left_hist_count * H_S_left + right_hist_count * H_S_right) / total_count

        info_gain = H_S - H_after

        return -(info_gain)

    def segmenter(self, data, labels):
        numFeatures = data.shape[1]

        bestImpurity = float('inf')
        bestFeature = None # index of best feature
        bestThreshold = None # the elements with the bestThreshold value are in the right-hand set

        for i in np.arange(numFeatures):
            left_label_hist = np.array([0, 0])
            right_label_hist = np.array([labels.shape[0] - labels.sum(), labels.sum()])

            if self.isBinaryFeature[i]: # continuous feature
                sorted_vals = np.sort(data[:, i])
                indices = np.argsort(data[:, i])
                for j in np.arange(sorted_vals.shape[0]):
                    if j != 0:
                        label_j1 = labels[indices[j-1]] # label of (j-1)th elem in the sorted_vals array
                        left_label_hist[label_j1] += 1
                        right_label_hist[label_j1] -= 1

                    imp = DecisionTree.impurity(left_label_hist, right_label_hist)
                    if imp < bestImpurity:
                        bestImpurity = imp
                        bestFeature = i
                        bestThreshold = sorted_vals[j]
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
        self.rootNode = self.growTree(data, labels)


    def growTree(self, data, labels):
        # Based on the GrowTree method described in lecture notes
        isPure, label = DecisionTree.pureSet(labels)
        if isPure:
            return Node(split_rule=None, left=None, right=None, label=label)
        else:
            split_rule = self.segmenter(data, labels)
            left_data, left_labels, right_data, right_labels = DecisionTree.split(split_rule, data, labels)
            return Node(split_rule, self.train(left_data, left_labels), self.train(right_data, right_labels))


    def predict(self, data):
        predicted_labels = np.zeros((data.shape[0]))
        for i in data.shape[0]:
            node = self.rootNode
            while node.label is not None:
                splitFeature, splitThreshold = node.split_rule
                if data[i, splitFeature] < splitThreshold:
                    node = node.left
                else:
                    node = node.right
            predicted_labels[i] = node.label
