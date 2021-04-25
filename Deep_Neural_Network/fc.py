import numpy as np

#勾配はクラス内で計算、勾配を用いた更新はoptimizerで行う
class FC:
    """
    ノード数n_nodes1からn_nodes2への全結合層
    Parameters
    ----------
    n_nodes1 : int
      前の層のノード数
    n_nodes2 : int
      後の層のノード数
    initializer : 初期化方法のインスタンス
    optimizer : 最適化手法のインスタンス
    """
    def __init__(self, n_nodes1, n_nodes2, activation,initializer, optimizer):
        self.optimizer = optimizer
        self.n_nodes1 = n_nodes1
        self.n_nodes2 = n_nodes2
        self.activation = activation
        # 初期化
        # initializerのメソッドを使い、self.Wとself.Bを初期化する
        self.W = initializer.W(self.n_nodes1,self.n_nodes2)
        self.B = initializer.B(self.n_nodes2)
    def forward(self, X):
        """
        フォワード
        Parameters
        ----------
        X : 次の形のndarray, shape (batch_size, n_nodes1)
            入力
        Returns
        ----------
        A : 次の形のndarray, shape (batch_size, n_nodes2)
            出力
        """
        #backwardでself.dWを求める時に使うためクラス変数化
        self.X = X
        A = self.X@self.W + self.B
        Z = self.activation.forward(A)
        return Z
    def backward(self, dZ,Y=None):
        """
        バックワード
        Parameters
        ----------
        dA : 次の形のndarray, shape (batch_size, n_nodes2)
            後ろから流れてきた勾配
        Returns
        ----------
        dZ : 次の形のndarray, shape (batch_size, n_nodes1)
            前に流す勾配
        """
        #活性化関数がSoftmaxの時だけはdZとYを渡す
        if Y is None:
            dA = self.activation.backward(dZ)
        else:
            dA = self.activation.backward(dZ,Y)
        #バイアスと重みの勾配の計算
        self.dB = np.mean(dA,axis=0)
        self.dW = (self.X.T@dA)/len(self.X)  #ここでXで割ってなかったせいで勾配が大きくなっていた
        dZ = dA@self.W.T
        #更新
        self = self.optimizer.update(self,dA)

        return dZ
