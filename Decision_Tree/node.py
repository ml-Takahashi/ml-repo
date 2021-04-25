import numpy as np
from calc_gini import calc_gini2

class Node2():
    def __init__(self,data,labels,parent = None,left=None,right=None,depth=0):
        self.data = data
        self.labels = labels
        self.left = left
        self.right = right
        self.depth = depth
        self.gini = calc_gini2(self)
        self.leaf_list = np.array([])

    def classify(self,threshold):
        left_index = np.where(self.data[:,self.depth] < threshold)[0]
        right_index = np.where(self.data[:,self.depth] >= threshold)[0]
        self.left = self.data[left_index],self.labels[left_index]
        self.right = self.data[right_index],self.labels[right_index]

    def set_node(self):
        left = self.left
        right = self.right
        self.left_node = Node2(left[0],left[1],parent=self,depth=self.depth+1)
        self.right_node = Node2(right[0],right[1],parent=self,depth=self.depth+1)

    def calc_info_gain(self):
        parent = self
        left = parent.left_node
        right = parent.right_node
        if len(parent.labels)==0:
            total_gini = parent.gini
        else:
            #print(left.gini)
            total_gini = (len(left.labels)/len(parent.labels))*left.gini + (len(right.labels)/len(parent.labels))*right.gini
        return parent.gini - total_gini
