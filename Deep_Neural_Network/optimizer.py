import numpy as np

class SGD:
    """
    確率的勾配降下法
    Parameters
    ----------
    lr : 学習率
    """
    def __init__(self, lr):
        self.lr = lr
    def update(self, layer,dA):
        """
        ある層の重みやバイアスの更新
        Parameters
        ----------
        layer : 更新前の層のインスタンス
        """
        #更新
        layer.W = layer.W - self.lr*layer.dW
        layer.B = layer.B - self.lr*layer.dB
        return layer

class AdaGrad:
    def __init__(self, lr):
        self.lr = lr
        self.H_w = 0
        self.H_b = 0
    def update(self, layer,dA):
        """
        ある層の重みやバイアスの更新
        Parameters
        ----------
        layer : 更新前の層のインスタンス
        """
        #重みの更新
        self.H_w += layer.dW**2 + 1e-7  #オーバーフロー対策
        layer.W -= self.lr*np.mean(layer.dW,axis=0)/(np.sqrt(self.H_w))

        #バイアスの更新
        self.H_b += (layer.dB)**2 + 1e-7  #オーバーフロー対策
        layer.B -= self.lr*np.mean(layer.dB,axis=0)/(np.sqrt(self.H_b))
        return layer
