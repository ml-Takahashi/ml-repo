import numpy as np
from node import Node2
from information_gain import information_gain2
from calc_gini import calc_gini2
import scipy


class ScratchDecesionTreeClassifierDepth2():
    """
    深さ2の決定木分類器のスクラッチ実装

    Parameters
    ----------
    verbose : bool
      学習過程を出力する場合はTrue
    """
    def __init__(self, verbose=False):
        # ハイパーパラメータを属性として記録
        self.verbose = verbose
    def fit(self, X, y):
        """
        決定木分類器を学習する
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            訓練データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            訓練データの正解値
        """
        self.y = y
        self.best_threshold = 0
        self.gain = 0
        threshold = []
        self.class_list = np.arange(2**X.shape[1])
        self.combination = np.array([])
        node = Node2(X,y)  #親ノードを生成
        count = 0
        for x_k in X[:,2]:
            for x_j in X[:,1]:
                for x_i in X[:,0]:
                    threshold = [x_i,x_j,x_k]
                    node.classify(threshold[0])  #深さ0のノードで分類
                    node.set_node()  #深さ１のノードへ分割して入れる
                    node.left_node.classify(threshold[1])  #深さ１の左ノードで分類
                    node.right_node.classify(threshold[1])  #深さ１の右ノードで分類
                    node.left_node.set_node()  #深さ2のノードへ分割して入れる
                    node.right_node.set_node()  #深さ2のノードへ分割して入れる

                    #深さ２のそれぞれのノードで分類
                    node.left_node.left_node.classify(threshold[2])
                    node.left_node.right_node.classify(threshold[2])
                    node.right_node.left_node.classify(threshold[2])
                    node.right_node.right_node.classify(threshold[2])

                    gain_value = node.left_node.calc_info_gain() + node.right_node.calc_info_gain()  #情報利得を取得
                    if self.gain < gain_value:
                        self.gain = gain_value
                        self.best_threshold = threshold
                        if self.verbose:
                            #ノードの中身を表示
                            print("threshold:",self.best_threshold)
                            print("depth1 left left left labels:",node.left_node.left_node.left[1])
                            print("depth1 left left right labels",node.left_node.left_node.right[1],end="\n\n")

                            print("depth1 left right left labels:",node.left_node.right_node.left[1])
                            print("depth1 left right right labels",node.left_node.right_node.right[1],end="\n\n")

                            print("depth1 right left left labels:",node.right_node.left_node.left[1])
                            print("depth1 right left right labels",node.right_node.left_node.right[1],end="\n\n")

                            print("depth1 right right left labels:",node.right_node.right_node.left[1])
                            print("depth1 right right right labels",node.right_node.right_node.right[1],end="\n\n")
                            print("------------------------------------------------------------------------")
                        self.class_list[0] = self.calc_mode(node.left_node.left_node.left[1])
                        self.class_list[1] = self.calc_mode(node.left_node.left_node.right[1])
                        self.class_list[2] = self.calc_mode(node.left_node.right_node.left[1])
                        self.class_list[3] = self.calc_mode(node.left_node.right_node.right[1])
                        self.class_list[4] = self.calc_mode(node.right_node.left_node.left[1])
                        self.class_list[5] = self.calc_mode(node.right_node.left_node.right[1])
                        self.class_list[6] = self.calc_mode(node.right_node.right_node.left[1])
                        self.class_list[7] = self.calc_mode(node.right_node.right_node.right[1])

    def calc_mode(self,X):  #決定木のラベル分けで使用
        if len(X)==0:
            return np.min(self.y)  #要素がないときはラベルの最小値を返す
        else:
            return scipy.stats.mode(X)[0]

    def predict(self, X):
        """
        決定木分類器を使いラベルを推定する
        """
        left,left_left,left_right,right,right_left,right_right = [],[],[],[],[],[]
        y_pred = np.empty(X.shape[0])
        left = np.where(X[:,0] < self.best_threshold[0])[0]
        right = np.where(X[:,0] >= self.best_threshold[0])[0]
        for i in left:
            if X[i,1] < self.best_threshold[1]:
                left_left.append(i)
            else:
                left_right.append(i)
        for i in right:
            if X[i,1] < self.best_threshold[1]:
                right_left.append(i)
            else:
                right_right.append(i)
        y_pred[left_left] = np.where(X[left_left,2] < self.best_threshold[2],self.class_list[0],self.class_list[1])
        y_pred[left_right] = np.where(X[left_right,2] < self.best_threshold[2],self.class_list[2],self.class_list[3])
        y_pred[right_left] = np.where(X[right_left,2] < self.best_threshold[2],self.class_list[4],self.class_list[5])
        y_pred[right_right] = np.where(X[right_right,2] < self.best_threshold[2],self.class_list[6],self.class_list[7])
        return y_pred
