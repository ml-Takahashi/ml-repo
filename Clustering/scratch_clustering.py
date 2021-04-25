import numpy as np
from functions import choose_center,SSE,assign_class,move_center

class ScratchKMeans():
    """
    K-meansのスクラッチ実装

    Parameters
    ----------
    n_clusters : int
      クラスタ数
    n_init : int
      中心点の初期値を何回変えて計算するか
    max_iter : int
      1回の計算で最大何イテレーションするか
    tol : float
      イテレーションを終了する基準となる中心点と重心の許容誤差
    verbose : bool
      学習過程を出力する場合はTrue
    """
    def __init__(self, n_clusters=4, n_init=100, max_iter=100, tol=1, verbose=False):
        # ハイパーパラメータを属性として記録
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
    def fit(self, X):
        """
        K-meansによるクラスタリングを計算
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            訓練データの特徴量
        """
        distance = np.full(self.max_iter*self.n_init,10000).reshape(self.max_iter,self.n_init)
        self.true_labels = np.empty(X.shape[0])
        self.true_center = np.empty(self.n_clusters*X.shape[1]).reshape(self.n_clusters,X.shape[1])
        self.best_sse = 100**100
        i = 0
        for j in range(self.n_init):
            #終了条件
            if distance[i,j] < self.tol:
                break
            #重心の初期化
            center = choose_center(X,self.n_clusters)
            #グルーピングし、クラスタ作成
            temp_class = assign_class(X,center)
            for i in range(self.max_iter):
                #平均値を求め、そこを重心とする
                center = move_center(X,temp_class,center)
                #全てのサンプルのデータ点との距離を計算する
                distance[i,j] = SSE(X,temp_class,center)
                #print("min.distance:",np.min(distance))
                #print(distance[i,j])
                #print("-----------------------")
                if self.best_sse > distance[i,j]:
                    self.true_labels = temp_class
                    self.true_center = center
                    self.best_sse = distance[i,j]
                    #print("更新")
                #終了条件
                if distance[i-1,j]==distance[i,j]:
                        #print("owari")
                        break
        if self.verbose:
            #verboseをTrueにした際は学習過程を出力
            print()
        pass
    def predict(self, X):
        """
        入力されたデータがどのクラスタに属するかを計算
        """
        return assign_class(X,self.true_center)
