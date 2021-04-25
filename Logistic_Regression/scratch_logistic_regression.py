import numpy as np

def loss_func(h_x,y,theta,lam):
    a = -y*np.log(h_x)
    b = (1-y)*np.log(1-h_x)
    c = (lam/(2*len(h_x)))*np.sum(theta[1:]**2)
    return (np.sum(a-b)/len(h_x)) + c

class ScratchLogisticRegression():
    """
    ロジスティック回帰のスクラッチ実装

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    no_bias : bool
      バイアス項を入れない場合はTrue
    verbose : bool
      学習過程を出力する場合はTrue

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      訓練データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証データに対する損失の記録

    """
    def __init__(self,num_iter=1000,lam=1e-3,lr=1e-2,bias=True, verbose=False,name="empty"):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose
        self.lam = lam
        #すでに使ったことのあるクラスだった場合にnameにクラス名をstr型で指定すると,
        #filesにあるバイナリファイルからすでに計算済みのθを取ってくる.
        save_filename = "files/"+name+".npz"
        #バイナリファイルからndarray読み込み
        if name == "empty":
            self.theta = np.empty(1)
        else:
            self.theta = np.load(save_filename)[name]
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
    def fit(self, X, y, X_val=np.empty(1), y_val=np.empty(1)):
        """
        ロジスティック回帰を学習する。検証データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

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
        #訓練用データ、検証用データの形を整える
        if y.ndim == 1:  #yが１次元のとき
            self.y = y[:,np.newaxis]  #_linear_hypothesisの返り値（行列）にサイズを合わせるためreshapeする
        else:
            self.y = y
        self.y_val = y_val
        if self.bias == True:  #バイアス項がある時
            if X.ndim == 1:  #１次元の時(２次元であればここはスルー)
                X = X[:,np.newaxis]  #バイアス項と結合できるようにXを２次元に変形
            self.X = np.concatenate([np.ones(X.shape[0])[:,np.newaxis],X],axis=1)  #バイアス項を結合
            if(len(X_val) != 1) & (len(y_val) != 1): #検証データが入力された場合（入力されない場合は何もしない）
                if X_val.ndim == 1:  #検証データが１次元の時
                    X_val = X_val[:,np.newaxis]  #バイアス項と結合できるようにXを２次元に変形
                self.X_val = np.concatenate([np.ones(X_val.shape[0])[:,np.newaxis],X_val],axis=1)  #バイアス項を結合

        else:  #バイアス項がない時
            if X.ndim ==1:  #Xが１次元の時
                self.X = X[:,np.newaxis]
                self.X_val = X_val[:,np.newaxis]
            else:  #Xが２次元の時
                self.X = X
                self.X_val = X_val

        #ここから計算
        if (len(X_val) != 1) & (len(y_val) != 1):  #検証データが入力された場合
            error = self._gradient_descent(self.X)
            self.loss[0] = loss_func(self.predict(self.X),y,self.theta,self.lam)
            self.val_loss[0] = loss_func(self.predict(self.X_val),y_val,self.theta,self.lam)
            for i in range(self.iter-1):  #iter-1回,_gradient_descentを実行する
                error = self._gradient_descent(self.X,error)
                self.loss[i+1] = loss_func(self.predict_proba(self.X),self.y,self.theta,self.lam)
                self.val_loss[i+1] = loss_func(self.predict_proba(self.X_val),self.y_val,self.theta,self.lam)
                if self.verbose:
                    #verboseをTrueにした際は学習過程を出力
                    print("{}回目".format(i+1))
                    print("theta:{}".format(self.theta))
                    print("損失:{}".format(self.val_loss[i+1]))
                    print("精度（平均2乗誤差）:{}\n".format(MSE(self.predict(self.X_val),self.y_val)))
        else:  #検証データが入力されない場合
            error = self._gradient_descent(self.X)
            self.loss[0] = loss_func(self.predict(self.X),y,self.theta,self.lam)
            self.val_loss[0] = loss_func(self.predict(self.X_val),y_val,self.theta,self.lam)
            for i in range(self.iter-1):
                error = self._gradient_descent(self.X,error)
                self.loss[i+1] = loss_func(self.predict_proba(self.X),self.y,self.theta,self.lam)
                self.val_loss[i+1] = loss_func(self.predict_proba(self.X_val),self.y_val,self.theta,self.lam)
                if self.verbose:
                    #verboseをTrueにした際は学習過程を出力
                    print("{}回目".format(i+1))
                    print("theta:\n{}".format(self.theta))

    def _gradient_descent(self, X,error=np.empty(1)):
        """
        θとXを用いて次のθを計算する
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            訓練データの特徴量

        error:次の形のndarray,shape (n_samples,)
            予測値と正解値の残差
        --------
        Returns
        --------
        次の形のndarray,shape(n_samples,)
        予測値と正解値の残差
        """
        if error.shape == (1,):  #errorが渡されていない時
            #θが与えられていない時だけ初期化する
            if self.theta.shape == (1,):
                self.theta = np.ones(X.shape[1]).reshape(1,-1)  #θを1で初期化
            error = (self._linear_hypothesis(X) - self.y).reshape(1,-1)
        else:  #errorが渡されている時
            error = (self.predict_proba(X) - self.y).flatten()
        if self.bias == True:  #バイアス項がある場合
            self.theta[0,0] = self.theta[0,0]- (self.lr/X.shape[0])*error@self.X[:,0]
            self.theta[0,1:] = self.theta[0,1:] - (self.lr/X.shape[0])*error@self.X[:,1:]+ (self.lam/X.shape[0])*self.theta[0,1:]
        else:  #バイアス項がない場合
            self.theta = self.theta - (self.lr/X.shape[0])*error@self.X+ (self.lam/X.shape[0])*self.theta
        return error

    def _linear_hypothesis(self, X):
        """
        線形の仮定関数を計算する
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
        訓練データ
        Returns
        -------
        次の形のndarray, shape (n_samples, 1)
        線形の仮定関数による推定結果
        """
        if X.ndim == 1:  #Xが１次元の時
            X = X[:,np.newaxis]  #２次元に変形
        if (X.shape[1] != self.theta.T.shape[0])&(self.bias==True):
            #x_0が含まれていない状態でXを受け取り（外部からのメソッド呼び出しを想定）、かつバイアス項を使う時
            X = np.concatenate([np.ones(X.shape[0])[:,np.newaxis],X],axis=1)
        return np.dot(X,self.theta.T)

    def predict(self, X):
        """
        ロジスティック回帰を使いラベルを推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            ロジスティック回帰による推定結果
        """
        #閾値を設定
        threshold = 0.5
        X_pred = self.predict_proba(X).flatten()
        X_pred[X_pred<threshold]  = 0
        X_pred[X_pred>=threshold] = 1
        return X_pred
    def predict_proba(self, X):
        """
        ロジスティック回帰を使い確率を推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            ロジスティック回帰による推定結果
        """

        return 1/(1+np.exp(-self._linear_hypothesis(X)))
