# -*- coding: utf-8 -*-
# This script implements the bilevel SGD method for SVM hyperparameter tuning in this paper:
# Couellan N, Wang W. Bi-level stochastic gradient for large scale support vector machine. Neurocomputing 2015;153:300–8. doi:10.1016/J.NEUCOM.2014.11.025
# Author: Wei Jiang
# Date: 1/18/2018
import numpy as np
import random as rd
import pandas as pd
from dynamic_plot import dynamic_plot



class bilevel_SGD():

    def __init__(self):
        self.C = None
        self.alpha = None
        self.beta = np.random.uniform(-1,1)
        self.C_min = 1e-4
        self.C_max = 1e6
        self.t_max = 2000 # maximal number of iterations
        self.lr_beta = 0.001 # learning rate (step size) for beta
        self.lr_C = 1 # learning rate for C

        self.accuracy_threshold = 0.97

    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        params:
            X_train: array, training data, each row is a sample
            y_train: 1d array, prediction label
            X_valid: array, validation data
            y_valid: 1d array, prediction label from validation data
        """

        ## initialize parameters
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.C = self.C_min
        feature_size = X_train.shape[1]

        X_t_size = X_train.shape[0]
        X_v_size = X_valid.shape[0]
        self.beta = np.random.uniform(-1, 1, feature_size)

        t = 1
        # print self.stop()
        dp = dynamic_plot(xlim=(0,self.t_max), ylim=(0, 1), xlabel = 'Iteration', ylabel = 'error')

        while (self.stop() < self.accuracy_threshold) and (t <= self.t_max):

            error_vector_t = np.multiply(y_train, np.dot(X_train, self.beta))
            error_vector_v = np.multiply(y_valid, np.dot(X_valid, self.beta))

            l = np.random.choice(np.where(error_vector_t<1)[0])
            p = np.random.choice(np.where(error_vector_v<1)[0])

            # update lower level gradient
            G_grad =  self.beta - self.C*y_train[l]*X_train[l,]

            # update upper level gradient
            C_grad = - np.dot(y_valid[p]*X_valid[p,], y_train[l]*X_train[l,])

            # print C_grad
            
            self.beta = self.beta - self.lr_beta*G_grad
            # print self.beta

            self.lr_C = 1/np.absolute(t*np.sqrt(feature_size)*C_grad)
            self.C = self.C - self.lr_C*C_grad

            if self.C < self.C_min:
                self.C = self.C_min
            if self.C > self.C_max:
                self.C = self.C_max

            # print self.stop()
            # print self.C
            dp.update_line(t, self.stop())

            t += 1
        # print 'final accuracy: ', self.stop()
        dp.fig.savefig('svmguide1_error_profile_stepC_1_stepW_0.001_one_validation.png')

        print 'final C: ', self.C
        print 'final cross-val accuracy: ', self.stop()


    def stop(self):
        """
        return: True if stoping criteria satisfied, otherwise False
        """
        pred_v = self.predict(self.X_valid)

        accuracy_v = self.accuracy(self.y_valid, pred_v)
        return accuracy_v 



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

def svmguide1():
    df = pd.read_csv("data/svmguide1.csv", header=None)
    # print df.head()

    X = df.values[:,range(1,5)]
    y = df.values[:,0]

    X = X[:, np.var(X, axis=0)>0]

    X = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x))*2 -1, axis=0 , arr = X)

    y[y==0] = -1
    y[y==1] = 1
    return X, y


if __name__ == '__main__':
    # X = pd.read_csv('../OptimizationProject_Wei/adult_x.csv', header=None)
    # y = pd.read_csv('../OptimizationProject_Wei/adult_y.csv', header=None)
    X, y = svmguide1()

    temp_ind = np.random.randint(X.shape[0], size=X.shape[0]/2)
    val_ind = list(set(range(0, X.shape[0])) - set(temp_ind))
    X_train = X[temp_ind,]
    y_train = y[temp_ind,]

    X_valid = X[val_ind, ]
    y_valid = y[val_ind, ]


    bi_SGD = bilevel_SGD()
    bi_SGD.fit(X_train, y_train, X_valid, y_valid)
