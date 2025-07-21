import math
from collections import Counter

class DesTree:
    def __init__(self):
        self.tree = None
        self.default_class = None
    
    def entropy(self, data):
        labels = [row["label"] for row in data]
        counts = Counter(labels)
        total = len(labels)
        entropy = 0.0
        for count in counts.values():
            p = count/total
            entropy -= p*math.log2(p)
        return entropy
    
    def info_gain(self, data, feature, threshold):
        total_entropy = self.entropy(data)
        total_len = len(data)

        subset1 = [row for row in data if row[feature]<=threshold]
        subset2 = [row for row in data if row[feature]>threshold]

        entropy1 = self.entropy(subset1) if subset1 else 0
        entropy2 = self.entropy(subset2) if subset2 else 0

        weighted_entropy = (len(subset1) / total_len) * entropy1 + (len(subset2) / total_len) * entropy2
        gain = total_entropy - weighted_entropy
        return gain 
    
    def find_best_split(self, data):
        best_gain = -1
        best_threshold = None
        best_feature = None

        for feature in data[0]:
            if feature == "label": continue
            values = sorted(set(row[feature] for row in data))
            for threshold in values:
                gain = self.info_gain(data, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_feature = feature
        return best_gain, best_threshold, best_feature
    
    def build_tree(self, data):
        labels = [row["label"] for row in data]
        if len(set(labels)) == 1:
            return Node(label = labels[0])
        if not data:
            return Node(label = self.default_class)
        best_gain, best_threshold, best_feature = self.find_best_split(data)
        if best_gain == 0 or best_feature == 0:
            majority_label = Counter(labels).most_common(1)[0][0]
            return Node(label=majority_label)
        left_split = [row for row in data if row[best_feature] <= best_threshold]
        right_split = [row for row in data if row[best_feature] > best_threshold]
        left_child = self.build_tree(left_split)
        right_child = self.build_tree(right_split)
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    def predict_one(self, row, node = None):
        if node is None:
            node = self.tree
        if node.label is not None:
            return node.label
        if row[node.feature] <= node.threshold: 
            return self.predict_one(row, node.left)
        else:
            return self.predict_one(row, node.right)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label
