
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import feature_extraction


class Node(object):
    def __init__(self, split_rule, left, right, label):
        self.split_rule = split_rule
        self.left = left
        self.right = right
        self.label = label

class DecisionTree(object):
    def __init__(self):

        self.binaryFeatures = []

    def impurity(self, left_label_hist, right_label_hist):
        """
        Calculates the impurity (badness) of a split. Uses entropy measure.
        :param left_label_hist: an array where key is 0 or 1, value is count
        :param right_label_hist:
        :return: impurity
        """
        total_count = left_label_hist[0] + right_label_hist[0] + left_label_hist[1] + right_label_hist[1]
        p_0_S = 1.0 * (left_label_hist[0] + right_label_hist[0]) / total_count #p_c is proportion of points in S that are in class C
        p_1_S = 1.0 * (left_label_hist[1] + right_label_hist[1]) / total_count
        H_S = -(p_0_S * np.log2(p_0_S) + p_1_S * np.log2(p_1_S)) # H(S), i.e. the entropy of S

        left_hist_count = (left_label_hist[0] + left_label_hist[1])
        p_0_S_left = 1.0 * left_label_hist[0] / left_hist_count
        p_1_S_left = 1.0 * left_label_hist[1] / left_hist_count
        H_S_left = -(p_0_S_left * np.log2(p_0_S_left) + p_1_S_left * np.log2(p_1_S_left))  # H(S_l), i.e. the entropy of S_l

        right_hist_count = (right_label_hist[0] + right_label_hist[1])
        p_0_S_right = 1.0 * right_label_hist[0] / right_hist_count
        p_1_S_right = 1.0 * right_label_hist[1] / right_hist_count
        H_S_right = -(p_0_S_right * np.log2(p_0_S_right) + p_1_S_right * np.log2(p_1_S_right))  # H(S_l), i.e. the entropy of S_l

        H_after = (left_hist_count * H_S_left + right_hist_count * H_S_right) / total_count

        info_gain = H_S - H_after

        return -(info_gain)

    def segmenter(self, data, labels):
        numFeatures = data.shape[1]

        bestImpurity = float('inf')
        bestFeature = NULL # index of best feature
        bestThreshold = NULL # the elements with the bestThreshold value are in the right-hand set

        for i in np.arange(numFeatures):
            if i not in binaryFeatures: # continuous feature
                left_label_hist = np.array([0, 0])
                right_label_hist = np.array([labels.shape[0] - labels.sum(), labels.sum()])

                # sorted_vals, counts = np.unique(data[:, i], return_counts=True)
                # counts_1 =  # counts (that have label of 1)

                sorted_vals = np.sort(data[:, i])
                indices = np.argsort(data[:, i])
                for j in np.arange(sorted_vals.shape[0] + 1):
                    if j != 0:
                        label_j1 = label[indices[j-1]] # label of (j-1)th elem in the sorted_vals array
                        left_label_hist[label_j1] = left_label_hist[label_j1] + 1
                        right_label_hist[label_j1] = right_label_hist[label_j1] - 1

                    imp = impurity(left_label_hist, right_label_hist)
                    if imp < bestImpurity:
                        bestImpurity = imp
                        bestFeature = i
                        if j == sorted_vals.shape[0]:
                            bestThreshold = sorted_vals[j-1] + 1
                        else:
                            bestThreshold = sorted_vals[j]
            else: # binary feature
                #@@@ CONTINUE HERE
                print 'Error: Binary feature functionality not implemented.'



        if bestFeature == NULL or bestThreshold == NULL:
            print 'Error: in segmenter(.), bestFeature or bestThreshold were not found.'

        return bestFeature, bestThreshold





    def train(self, data, labels):


    def predict(self, data):