import numpy as np

class Tanh:
    def __init__(self):
        pass

    def forward(self,A):
        self.A = A
        Z = np.tanh(self.A)
        return Z

    def backward(self,dZ):
        dA = dZ*(1 - np.tanh(self.A**2))
        return dA

class Sigmoid:
    def __init__(self):
        pass

    def forward(self,A):
        self.A = A
        self.Z = 1/(1 + np.exp(-self.A))
        return self.Z

    def backward(self,dZ):
        dA = dZ*(1 - self.Z)*self.Z
        return dA

class Softmax:
    def __init__(self):
        pass

    def forward(self,A):
        A = A - np.max(A,axis=0)  #オーバーフロー対策
        Z = np.exp(A)/np.sum(np.exp(A),axis=1).reshape(-1,1)
        return Z

    def backward(self,Z,Y):
        dA = Z - Y
        return dA

class ReLU:
    def __init__(self):
        pass

    def forward(self,A):
        self.A = A
        Z = np.maximum(self.A,0)
        return Z

    def backward(self,dZ):
        dA = np.where(self.A > 0,dZ,0)
        return dA
