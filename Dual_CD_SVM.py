# This script implments the dual coordinate descent method for SVM in this paper:
# Hsieh, Cho-Jui, et al. "A dual coordinate descent method for large-scale linear SVM." Proceedings of the 25th international conference on Machine learning. ACM, 2008.

# Author: Wei Jiang
# Date: 1/12/2018

import numpy as np
import random as rd
import pandas as pd

class DCD_SVM():

    def __init__(self, C):
        self.alpha = None # dual variables
        self.beta = None # primal vaiables
        self.C = C

    def fit(self, X, y):
        """
        params:
            X: array, each row is a sample
            y: 1d array, prediction label
        """
        self.X = X
        self.y = y
        X_size = X.shape[0]
        Q_ii = (X*X).sum(axis=1)

        # temp_Q = np.dot(y, X)
        # self.Q = np.dot(temp_Q, np.transpose(temp_Q))

        self.alpha = np.zeros(X_size) # initialize alpha

        self.beta = np.dot(np.multiply(self.alpha, y), X)

        i = 0
        while not self._check_optimality() and i < 100000:
            i += 1
            ind = rd.randint(0, X_size-1)

            temp_alpha = self.alpha[ind]
            Gradient = y[ind]* np.dot(self.beta, X[ind,]) - 1

            # print Gradient

            if temp_alpha == 0:
                Proj_Gradient = min(Gradient, 0)
            elif temp_alpha == self.C:
                Proj_Gradient = max(Gradient, 0)
            elif 0 < temp_alpha < self.C:
                Proj_Gradient = Gradient

            if not np.isclose(Proj_Gradient, 1e-9):
                self.alpha[ind] = min(max(temp_alpha - Gradient/ Q_ii[ind], 0), self.C)
                self.beta = self.beta + y[ind]*(self.alpha[ind] - temp_alpha)*X[ind,]

    def _check_optimality(self):
        """
        Check optimality condition using duality gap
        """

        dual_obj = -0.5* np.dot(self.beta, self.beta) + np.sum(self.alpha)

        prim_obj = 0.5* np.dot(self.beta, self.beta) + self.C * np.sum( np.maximum(1 - np.multiply(np.dot(self.X, self.beta), self.y), 0))

        # print (prim_obj - dual_obj)
        self.gap = prim_obj - dual_obj
        if self.gap <= 1e-6:
            return True
        else:
            return False

    def predict(self):
        return None

    def accuracy(self):
        correct_cases = np.multiply(np.dot(self.X, self.beta), self.y) > 0

        return sum(correct_cases)*1.0/len(self.y)


if __name__ == '__main__':
    X = pd.read_csv('../OptimizationProject_Wei/adult_x.csv',header=None)
    y = pd.read_csv('../OptimizationProject_Wei/adult_y.csv',header=None)

    X = X.values


    X = X[:, np.var(X, axis=0)>0]

    # X = (X - np.mean(X, axis = 0))/np.var(X, axis=0)
    X = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x)), axis =0 , arr = X)

    y = y.values.flatten()
    y[y==0] = -1
    y[y==1] = 1
    print X.shape

    svm = DCD_SVM(C=1)
    svm.fit(X,y)
    
    print svm.gap
    print svm.accuracy()