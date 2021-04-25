from sklearn.metrics import mean_squared_error

#損失関数
def loss_func(y_pred,y):
    return mean_squared_error(y_pred,y)/2
