import numpy as np

#clusterから中心点を選ぶ関数
def choose_center(cluster,K):
    index_list = []
    i = 0
    while i < K:
        num = int(np.random.randint(0,len(cluster)))
        if num not in index_list:
            index_list.append(num)
            i += 1
    return np.sort(cluster[index_list,:])

#クラスタ内誤差平方和を計算する関数
def SSE(X,labels,center):
    X = np.concatenate((X,labels),axis=1)
    #Xは０か１かわかるようにラベルをくっつけておく
    SSE = 0
    for x in X:
        for i,c in enumerate(center):
            if x[-1] == i:
                SSE += np.sum(x[:-1]-c)**2
    return SSE

#Xにクラス番号を割り当て、割り当てられたクラスを格納したを返す関数
def assign_class(X,center):
    class_list = np.empty(len(X))
    for j,x in enumerate(X):
        norm = np.empty(0)
        for i,c in enumerate(center):
            norm = np.append(norm,np.linalg.norm(x-c))
            if i == (len(center)-1):
                class_list[j] = np.argmin(norm)
    return class_list.reshape(-1,1)

#移動したクラスの座標を返す関数
def move_center(X,labels,center):
    X = np.concatenate((X,labels),axis=1)
    #Xは０か１かわかるようにラベルをくっつけておく
    center_position = np.empty(center.shape[0]*center.shape[1]).reshape(center.shape[0],center.shape[1])
    for i,c in enumerate(center):
        index = np.where(X[:,-1]==i)[0]
        if  len(index) == 0:
            center_position[i,:] = center[i]
        else:
            center_position[i,:] = np.mean(X[index,:-1],axis=0)
    return center_position
