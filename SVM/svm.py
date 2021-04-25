import numpy as np

class ScratchSVMClassifier():
    """
    SVM分類器のスクラッチ実装

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    kernel : str
      カーネルの種類。線形カーネル（linear）か多項式カーネル（polly）
    threshold : float
      サポートベクターを選ぶための閾値
    verbose : bool
      学習過程を出力する場合はTrue

    Attributes
    ----------
    self.n_support_vectors : int
      サポートベクターの数
    self.index_support_vectors : 次の形のndarray, shape (n_support_vectors,)
      サポートベクターのインデックス
    self.X_sv :  次の形のndarray, shape(n_support_vectors, n_features)
      サポートベクターの特徴量
    self.lam_sv :  次の形のndarray, shape(n_support_vectors, 1)
      サポートベクターの未定乗数
    self.y_sv :  次の形のndarray, shape(n_support_vectors, 1)
      サポートベクターのラベル

    """
    def __init__(self, num_iter = 10, lr = 1e-2, kernel='linear', threshold=1e-5, verbose=False,gamma=1,theta=0,d=1):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.kernel = kernel
        self.threshold = threshold
        self.verbose = verbose
        self.gamma = gamma
        self.theta = theta
        self.d = d

    def _reshape_dim(self,x):
        if x is not None:  #xがNonではないとき
            if x.ndim == 1:
                #xが1次元ならば２次元に変換して返す
                return x.reshape(-1,1)
            else:
                return x
        else:
            return x

    def fit(self, X, y, X_val=None, y_val=None):
        """
        SVM分類器を学習する。検証データが入力された場合はそれに対する精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            訓練データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            訓練データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証データの正解値
        """
        #lamdaもyも(n,1)の２次元になるようにreshapeしておく
        self.lamda = np.full(X.shape[0],1e-7).reshape(-1,1)  #λを0.5で初期化
        #それぞれ１次元だった場合に２次元になるようにreshapeを行い,Noneの場合はそのままNoneが入る
        X = self._reshape_dim(X)
        y = self._reshape_dim(y)
        X_val = self._reshape_dim(X_val)
        y_val = self._reshape_dim(y_val)
        self.label0 = np.min(y)
        self.label1 = np.max(y)


        if (X_val is not None) & (y_val is not None):  #検証用データがあるとき
            for i in range(self.iter):
                self._lagrange_multiplier(X,y)  #self.lamdaの更新
                self.support_vector_index = np.where(self.lamda > self.threshold)[0]  #サポートベクターのindexを選ぶ
                self.support_vector = X[self.support_vector_index]  #サポートベクターの決定
                self.support_vector_label = y[self.support_vector_index]  #サポートベクターのラベル
                if self.verbose:
                    #verboseをTrueにした際は学習過程を出力
                    print("サポートベクター:{}個".format(self.support_vector.shape[0]))
                    print("正解率:{}".format(accuracy_score(y_val,self.predict(X_val))))
        else:  #検証用データがないとき
            for i in range(self.iter):
                self._lagrange_multiplier(X,y)  #self.lamdaの更新
                self.support_vector_index = np.where(self.lamda > self.threshold)[0]  #サポートベクターのindexを選ぶ
                self.support_vector = X[self.support_vector_index]  #サポートベクターの決定
                self.support_vector_label = y[self.support_vector_index]  #サポートベクターのラベル
                if self.verbose:
                    #verboseをTrueにした際は学習過程を出力
                    print("サポートベクター:{}個".format(self.support_vector.shape[0]))

    def _lagrange_multiplier(self,x,y):
        for i in range(x.shape[0]):
            sigma = 0
            for j in range(x.shape[0]):
                sigma += np.sum(self.lamda[j] *(y[i]*y[j].T)*self._kernel_func(x[i],x[j]))  #ここを内積に変更
            self.lamda[i,0] = self.lamda[i,0] + self.lr*(1 - sigma)
            if self.lamda[i,0] < 0:
                self.lamda[i,0] = 0

    def _kernel_func(self,x,y):
            return (self.gamma*y@x.T + self.theta)**self.d

    def predict(self, X):
        """
        SVM分類器を使いラベルを推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            SVM分類器による推定結果
        """
        fx = np.empty(X.shape[0])
        for i,x in enumerate(X):
            x = x.reshape(1,-1)
            fx[i] = np.sum((self.lamda[self.support_vector_index]*self.support_vector_label).reshape(1,-1)@self._kernel_func(x,self.support_vector))
        return np.where(fx > 0,self.label1,self.label0)
