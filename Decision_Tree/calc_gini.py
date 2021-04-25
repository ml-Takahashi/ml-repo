import numpy as np

def calc_gini2(node):
    class_num = np.unique(node.labels,return_counts=True)[1]
    if class_num.shape[0] == 0:
        class_num = np.zeros(2)
    elif class_num.shape[0] == 1:
        class_num = np.append(class_num,0)
    if len(node.labels)==0:
        sigma = 0
    else:
        sigma = (class_num[0]/len(node.labels))**2 + (class_num[1]/len(node.labels))**2
    return 1- sigma
