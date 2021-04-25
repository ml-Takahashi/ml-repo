from calc_gini import calc_gini2

def information_gain2(parent):
    total_gini = (len(parent.left_node.labels)/len(parent.labels))*parent.left_node.gini + \
    (len(parent.right_node.labels)/len(parent.labels))*parent.right_node.gini
    return parent_gini - total_gini
