import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters(lenw):
    w = np.random.randn(1, lenw)
    b = 0
    return w, b


def forward_prop(X, w, b):   # w-->lnx, X-->nxm
    z = np.dot(w, X) + b     # z--> lxm n_b_vector = [b b b ...]
    return z


def cost_function(z, y):
    m = y.shape[1]
    J = (1/(2*m)) * np.sum(np.square(z-y))
    return J


def back_prop(X, y, z):
    m = y.shape[1]
    dz = (1/m)*(z-y)
    dw = np.dot(dz, X.T)   # dw --> lxx
    db = np.sum(dz)
    return dw, db


def gradient_descent_update(w, b, dw, db, learning_rate):
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b


def linear_regression_model(X_train, y_train, X_val, y_val, learning_rate, epochs):
    lenw = X_train.shape[0]
    w, b = initialize_parameters(lenw)

    costs_train = []
    m_train = y_train.shape[1]
    m_val = y_val.shape[1]

    for i in range(1, epochs + 1):
        z_train = forward_prop(X_train, w, b)
        cost_train = cost_function(z_train, y_train)
        dw, db = back_prop(X_train, y_train, z_train)
        w, b = gradient_descent_update(w, b, dw, db, learning_rate)

        if i % 10 == 0:
            costs_train.append(cost_train)
        MAE_train = (1/m_train) * np.sum(np.abs(z_train - y_train))

        z_val = forward_prop(X_val, w, b)
        cost_val = cost_function(z_val, y_val)
        MAE_val = (1/m_val) * np.sum(np.abs(z_val - y_val))

        print(f"Epochs {str(i)} / {str(epochs)}: ")
        print(f"Training cost {str(cost_train)} | Validation cost {str(cost_val)}")
        print(f"Training MAE {str(MAE_train)} | Validation MAE {str(MAE_val)}")

    plt.plot(costs_train)
    plt.xlabel('Iterations(per tens)')
    plt.ylabel('Training cost')
    plt.title(f'Learning rate {learning_rate}')
    plt.show()

# def r2_score(y_true, y_pred):
#     corr_matrix = np.corrcoef(y_true, y_pred)
#     corr = corr_matrix[0, 1]
#     return corr ** 2
#
#
# class LinearRegression:
#     def __init__(self, learning_rate=0.001, n_iters=1000):
#         self.lr = learning_rate
#         self.n_iters = n_iters
#         self.weights = None
#         self.bias = None
#
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#
#         # init parameters
#         self.weights = np.zeros(n_features)
#         self.bias = 0
#
#         # gradient descent
#         for _ in range(self.n_iters):
#             y_predicted = np.dot(X, self.weights) + self.bias
#             # compute gradients
#             dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
#             db = (1 / n_samples) * np.sum(y_predicted - y)
#
#             # update parameters
#             self.weights -= self.lr * dw
#             self.bias -= self.lr * db
#
#     def predict(self, X):
#         y_approximated = np.dot(X, self.weights) + self.bias
#         return y_approximated