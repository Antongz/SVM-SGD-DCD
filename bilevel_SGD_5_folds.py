# -*- coding: utf-8 -*-
# This script implements the bilevel SGD method for SVM hyperparameter tuning in this paper:
# Couellan N, Wang W. Bi-level stochastic gradient for large scale support vector machine. Neurocomputing 2015;153:300–8. doi:10.1016/J.NEUCOM.2014.11.025
# Author: Wei Jiang
# Date: 1/18/2018
import numpy as np
import random as rd
import pandas as pd


class bilevel_SGD():

    def __init__(self):
        self.C = None
        self.alpha = None
        self.beta = None
        self.C_min = 1e-4
        self.C_max = 1e6
        self.t_max = 1000 # maximal number of iterations
        self.lr_beta = 0.01 # learning rate (step size) for beta
        self.lr_C = 0.01 # learning rate for C

        self.accuracy_threshold = 0.97

    def fit(self, X, y, skf):
        """
        params:
            X: array, data, each row is a sample
            y: 1d array, prediction label
            skf: object, sklearn stratifiedKfold 
        """

        ## initialize parameters
        self.X = X
        self.y = y
        train_ind_ls = []
        test_ind_ls = []
        for train_ind, test_ind in skf.split(X, y):
            train_ind_ls.append(train_ind)
            test_ind_ls.append(test_ind)
        self.indice_gen = zip(train_ind_ls, test_ind_ls)
        

        self.C = self.C_min
        feature_size = X.shape[1]
        self.fold_num = skf.get_n_splits(self.X, self.y)

        self.beta = np.random.uniform(-1, 1, (self.fold_num, feature_size))

        t = 1
        print self.stop()
        while (self.stop() < self.accuracy_threshold) and (t <= self.t_max):
            C_grad_ls = []
            for i, index in enumerate(self.indice_gen):
                
                train_index = index[0]
                test_index = index[1]
                X_train, X_valid = X[train_index], X[test_index]
                y_train, y_valid = y[train_index], y[test_index]

                error_vector_t = np.multiply(y_train, np.dot(X_train, self.beta[i,]))
                error_vector_v = np.multiply(y_valid, np.dot(X_valid, self.beta[i,]))

                l = np.random.choice(np.where(error_vector_t<1)[0])
                p = np.random.choice(np.where(error_vector_v<1)[0])

                # update lower level gradient
                G_grad =  self.beta[i,] - self.C*y_train[l]*X_train[l,]
                self.beta[i,] = self.beta[i,] - self.lr_beta*G_grad

                # update upper level gradient
                C_grad_ls.append(- np.dot(y_valid[p]*X_valid[p,], y_train[l]*X_train[l,]))

            C_grad = sum(C_grad_ls)/self.fold_num

            # print C_grad
            
            # print self.beta

            self.C = self.C - self.lr_C*C_grad

            if self.C < self.C_min:
                self.C = self.C_min
            if self.C > self.C_max:
                self.C = self.C_max

            print self.stop()
            # print self.C

            t += 1


    def stop(self):
        """
        return: True if stoping criteria satisfied, otherwise False
        """
        accuracy_v_ls = []
        for i, index in enumerate(self.indice_gen):
            test_index = index[1]
            X_valid =  self.X[test_index]
            y_valid =  self.y[test_index]
            pred_v = np.sign(np.dot(X_valid, self.beta[i,]))

            accuracy_v_ls.append(self.accuracy(y_valid, pred_v))
        return sum(accuracy_v_ls)/self.fold_num


    def _duality_gap(self):
        dual_obj = -0.5* np.dot(self.beta, self.beta) + np.sum(self.alpha)

        prim_obj = 0.5* np.dot(self.beta, self.beta) + self.C * np.sum( np.maximum(1 - np.multiply(np.dot(self.X, self.beta), self.y), 0))

            # print (prim_obj - dual_obj)
        self.gap = prim_obj - dual_obj

    def predict(self, X):

        pred = np.sign(np.dot(X, self.beta))
        return pred

    def accuracy(self, y_true, y_pred):

        return sum(y_true==y_pred)*1.0/len(y_true)


if __name__ == '__main__':
    from sklearn.model_selection import StratifiedKFold
    # X = pd.read_csv('../OptimizationProject_Wei/adult_x.csv', header=None)
    # y = pd.read_csv('../OptimizationProject_Wei/adult_y.csv', header=None)
    df = pd.read_csv("data/breast-cancer-wisconsin.txt", na_values ='?', header=None)
    df = df.dropna(axis='index')
    print df.head()
    print df.dtypes

    X = df.values[:,range(1,10)]
    y = df.values[:,10]


    X = X[:, np.var(X, axis=0)>0]

    # X = (X - np.mean(X, axis = 0))/np.var(X, axis=0)
    X = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x))*2 -1, axis=0 , arr = X)

    print np.apply_along_axis(lambda x: np.mean(x), axis=0 , arr = X)

    # y = y.values.flatten()
    y[y==2] = -1
    y[y==4] = 1
    print X.shape

    skf = StratifiedKFold(n_splits=5)
    # print skf.split(X, y)
    


    bi_SGD = bilevel_SGD()
    bi_SGD.fit(X, y, skf)
