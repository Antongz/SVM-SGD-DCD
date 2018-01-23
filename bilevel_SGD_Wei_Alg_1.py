# -*- coding: utf-8 -*-
# This script implements the bilevel SGD method for SVM hyperparameter tuning in this paper:
# Couellan N, Wang W. Bi-level stochastic gradient for large scale support vector machine. Neurocomputing 2015;153:300–8. doi:10.1016/J.NEUCOM.2014.11.025
# Author: Wei Jiang
# Date: 1/18/2018
import numpy as np
import random as rd
import pandas as pd
from dynamic_plot import dynamic_plot
import matplotlib.pyplot as plt



class bilevel_SGD_Alg1():

    def __init__(self):
        self.C = None
        self.alpha = None
        self.beta = None
        self.C_min = 1e-4
        self.C_max = 1e6
        self.t_max = 20000 # maximal number of iterations
        self.lr_beta = 0.001 # learning rate (step size) for beta
        self.lr_C = 0.001 # learning rate for C

        self.accuracy_threshold = 0.97

        self.duality_gap_tol = 1e-3
        self.C_ls = []
        self.loss_ls = []
        self.accuracy_ls =[]

    def dual_CD_step(self):
        """
        Do dual coordinate descent on validation set.
        params:
            X: array, each row is a sample
            y: 1d array, prediction label
        """

        # temp_Q = np.dot(y, X)
        # self.Q = np.dot(temp_Q, np.transpose(temp_Q))

        if not self._check_optimality(method='duality_gap'): # if alpha is not optimal
            ind_permuted = np.random.permutation(self.X_t_size)
            for ind in ind_permuted:
                temp_alpha = self.alpha[ind]
                self.Gradient[ind] = self.y_train[ind]* np.dot(self.beta, self.X_train[ind,]) - 1

                # print self.Gradient[ind]

                if temp_alpha == 0:
                    self.Proj_grad[ind] = min(self.Gradient[ind], 0)
                elif temp_alpha == self.C:
                    self.Proj_grad[ind] = max(self.Gradient[ind], 0)
                elif 0 < temp_alpha < self.C:
                    self.Proj_grad[ind] = self.Gradient[ind]

                if not np.isclose(self.Proj_grad[ind], 1e-9):
                    self.alpha[ind] = min(max(temp_alpha - self.Gradient[ind]/ self.Q_ii[ind], 0), self.C)
                    self.beta = self.beta + self.y_train[ind]*(self.alpha[ind] - temp_alpha)*self.X_train[ind,]

    def _check_optimality(self, method):
        """
        Check optimality condition using duality gap
        params:
            method: str,'duality_gap'; 
                    'gradient': check if all projected gradient are close to 0
                    'gradient_gap': check if the difference between largest and smallest projected gradients are less than a tolerance
        """
        if method == 'duality_gap':
            self._duality_gap()
            # print self.gap
            
            if self.gap <= self.duality_gap_tol:

                return True
            else:
                return False
        elif method == 'gradient':
            return all(np.isclose(self.Proj_grad,0))
        elif method == 'gradient_gap':
            gap = max(self.Proj_grad) - min(self.Proj_grad)

            if gap <= self.gradient_gap_tol:
                return True
            else:
                return False

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

        self.M = np.multiply(self.y_train, np.transpose(self.X_train))

        self.C = self.C_min
        feature_size = X_train.shape[1]

        self.X_t_size = X_train.shape[0]
        self.X_v_size = X_valid.shape[0]

        # initialize for dual CD step
        self.Q_ii = (self.X_train*self.X_train).sum(axis=1)
        self.alpha = np.zeros(self.X_t_size) # initialize alpha
        # self.alpha = np.random.uniform(-0.1,0.1, self.X_t_size)

        self.Gradient = np.ones(self.X_t_size)
        self.Proj_grad = np.random.uniform(-1,1,self.X_t_size)

        self.beta = np.dot(np.multiply(self.alpha, self.y_train), self.X_train)

        t = 1
        # print self.stop()
        # dp = dynamic_plot(xlim=(0,self.t_max), ylim=(0, 1), xlabel = 'Iteration', ylabel = 'Accuracy')
        v = 0 # momentum update
        self.mu = 0.5
        while (self.stop() < self.accuracy_threshold) and (t <= self.t_max):

            # update lower level variables
            self.dual_CD_step()

            error_vector_t = np.multiply(y_train, np.dot(X_train, self.beta))
            error_vector_v = np.multiply(y_valid, np.dot(X_valid, self.beta))

            # l = np.random.choice(np.where(error_vector_t<1)[0])

            J_alpha = (self.alpha == self.C)

            if any(J_alpha):
                p = np.random.choice(np.where(error_vector_v<1)[0])

                # update upper level gradient
                C_grad = - np.dot(y_valid[p]*X_valid[p,], np.dot(self.M, J_alpha))

                # print C_grad

                # self.lr_C = 1/np.absolute(t*np.sqrt(feature_size)*C_grad)
                self.C = self.C - self.lr_C*C_grad

                # momentum update
                # v = self.mu*v - self.lr_C * C_grad
                # self.C += v


                if self.C < self.C_min:
                    self.C = self.C_min
                if self.C > self.C_max:
                    self.C = self.C_max

                # print self.stop()
                print self.C
                # dp.update_line(t, self.stop())

            self.accuracy_ls.append(self.stop())
            self.C_ls.append(self.C)
            self.loss_ls.append(self.loss_upper())

            t += 1
        # print 'final accuracy: ', self.stop()
        # dp.fig.savefig('pima_error_profile_stepC_1_stepW_0.001_one_validation_Alg2.png')

        print 'final C: ', self.C
        print 'final cross-val accuracy: ', self.stop()
        print t

        f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(18,8))
        ax1.plot(range(0, self.t_max), self.accuracy_ls)
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Iteration')

        ax2.plot(range(0, self.t_max), self.loss_ls)
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Iteration')

        ax3.plot(self.C_ls, self.loss_ls,'*-')
        ax3.set_ylabel('Loss')
        ax3.set_xlabel('C')
        f.suptitle('SVMGUIDE1 data, Algorithm 1')
        plt.show()

    def stop(self):
        """
        return: True if stoping criteria satisfied, otherwise False
        """
        pred_v = self.predict(self.X_valid)

        accuracy_v = self.accuracy(self.y_valid, pred_v)
        return accuracy_v 

    def loss_upper(self):
        
        loss = np.sum( np.maximum(1 - np.multiply(np.dot(self.X_valid, self.beta), self.y_valid), 0))
        return loss


    def _duality_gap(self):
        dual_obj = -0.5* np.dot(self.beta, self.beta) + np.sum(self.alpha)

        prim_obj = 0.5* np.dot(self.beta, self.beta) + self.C * np.sum( np.maximum(1 - np.multiply(np.dot(self.X_train, self.beta), self.y_train), 0))

            # print (prim_obj - dual_obj)
        self.gap = prim_obj - dual_obj

    def predict(self, X):

        pred = np.sign(np.dot(X, self.beta))
        return pred

    def accuracy(self, y_true, y_pred):

        return sum(y_true==y_pred)*1.0/len(y_true)

def pima_data():
    df = pd.read_csv("data/pima-indians-diabetes.txt")

    X = df.values[:,range(0,8)]
    y = df.values[:,8]

    X = X[:, np.var(X, axis=0)>0]

    X = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x))*2 -1, axis=0 , arr = X)

    y[y==0] = -1
    y[y==1] = 1
    return X, y

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


    bi_SGD = bilevel_SGD_Alg1()
    bi_SGD.fit(X_train, y_train, X_valid, y_valid)
