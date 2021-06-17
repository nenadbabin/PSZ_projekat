import numpy as np
import matplotlib.pyplot as plt


class MultipleLinearRegression:

    def __init__(self):
        # No instance Variables required
        pass

    def forward(self, X, y, W):
        """
        Parameters:
        X (array) : Independent Features
        y (array) : Dependent Features/ Target Variable
        W (array) : Weights

        Returns:
        loss (float) : Calculated Sqaured Error Loss for y and y_pred
        y_pred (array) : Predicted Target Variable
        """
        y_pred = sum(W * X)
        loss = ((y_pred - y) ** 2) / 2  # Loss = Squared Error, we introduce 1/2 for ease in the calculation
        return loss, y_pred

    def updateWeights(self, X, y_pred, y_true, W, alpha, index):
        """
        Parameters:
        X (array) : Independent Features
        y_pred (array) : Predicted Target Variable
        y_true (array) : Dependent Features/ Target Variable
        W (array) : Weights
        alpha (float) : learning rate
        index (int) : Index to fetch the corresponding values of W, X and y

        Returns:
        W (array) : Update Values of Weight
        """
        for i in range(X.shape[1]):
            # alpha = learning rate, rest of the RHS is derivative of loss function
            W[i] -= (alpha * (y_pred - y_true[index]) * X[index][i])
        return W

    def train(self, X, y, epochs=10, alpha=0.001, random_state=0):
        """
        Parameters:
        X (array) : Independent Feature
        y (array) : Dependent Features/ Target Variable
        epochs (int) : Number of epochs for training, default value is 10
        alpha (float) : learning rate, default value is 0.001

        Returns:
        y_pred (array) : Predicted Target Variable
        loss (float) : Calculated Sqaured Error Loss for y and y_pred
        """

        num_rows = X.shape[0]  # Number of Rows
        num_cols = X.shape[1]  # Number of Columns
        W = np.random.randn(1, num_cols) / np.sqrt(num_rows)  # Weight Initialization

        # Calculating Loss and Updating Weights
        train_loss = []
        num_epochs = []
        train_indices = [i for i in range(X.shape[0])]
        for j in range(epochs):
            cost = 0
            np.random.seed(random_state)
            np.random.shuffle(train_indices)
            for i in train_indices:
                loss, y_pred = self.forward(X[i], y[i], W[0])
                cost += loss
                W[0] = self.updateWeights(X, y_pred, y, W[0], alpha, i)
            train_loss.append(cost)
            num_epochs.append(j)
        return W[0], train_loss, num_epochs

    def test(self, X_test, y_test, W_trained):
        """
        Parameters:
        X_test (array) : Independent Features from the Test Set
        y_test (array) : Dependent Features/ Target Variable from the Test Set
        W_trained (array) : Trained Weights
        test_indices (list) : Index to fetch the corresponding values of W_trained,
                              X_test and y_test

        Returns:
        test_pred (list) : Predicted Target Variable
        test_loss (list) : Calculated Sqaured Error Loss for y and y_pred
        """
        test_pred = []
        test_loss = []
        test_indices = [i for i in range(X_test.shape[0])]
        for i in test_indices:
            loss, y_test_pred = self.forward(X_test[i], W_trained, y_test[i])
            test_pred.append(y_test_pred)
            test_loss.append(loss)
        return test_pred, test_loss

    def predict(self, W_trained, X_sample):
        prediction = sum(W_trained * X_sample)
        return prediction

    def plotLoss(self, loss, epochs):
        """
        Parameters:
        loss (list) : Calculated Sqaured Error Loss for y and y_pred
        epochs (list): Number of Epochs

        Returns: None
        Plots a graph of Loss vs Epochs
        """
        plt.plot(epochs, loss)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.title('Plot Loss')
        plt.show()
