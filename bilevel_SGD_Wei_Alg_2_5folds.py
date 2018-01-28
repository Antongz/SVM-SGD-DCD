# -*- coding: utf-8 -*-
# This script implements the modified (my algorithm 2.1) bilevel SGD method for SVM hyperparameter tuning in this paper:
# Couellan N, Wang W. Bi-level stochastic gradient for large scale support vector machine. Neurocomputing 2015;153:300–8. doi:10.1016/J.NEUCOM.2014.11.025
# Author: Wei Jiang
# Date: 1/22/2018
import numpy as np
import random as rd
import pandas as pd
from dynamic_plot import dynamic_plot
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from data_example import *



class bilevel_SGD_Alg2_5folds():

    def __init__(self):
        self.C = None
        self.alpha = None
        self.beta = None
        self.C_min = 1e-4
        self.C_max = 1e6
        self.t_max = 20 # maximal number of iterations
        self.lr_beta = 0.001 # learning rate (step size) for beta
        self.lr_C = 0.00001 # learning rate for C

        self.accuracy_threshold = 0.97

        self.duality_gap_tol = 1e-3
        self.C_ls = []
        self.loss_ls = []
        self.accuracy_ls =[]

    def dual_CD_step(self, fold_i, X_train, y_train):
        """
        Do dual coordinate descent on validation set.
        params:
            X: array, each row is a sample
            y: 1d array, prediction label
        """

        # temp_Q = np.dot(y, X)
        # self.Q = np.dot(temp_Q, np.transpose(temp_Q))

        if not self._check_optimality('duality_gap', fold_i, X_train, y_train): # if alpha is not optimal
            ind_permuted = np.random.permutation(X_train.shape[0])
            for ind in ind_permuted:
                temp_alpha = self.alpha[fold_i][ind]
                Gradient = y_train[ind]* np.dot(self.beta[fold_i,], X_train[ind,]) - 1

                # print self.Gradient[ind]
                if temp_alpha == 0:
                    # print 'check'
                    Proj_grad = min(Gradient, 0)
                elif temp_alpha >= self.C:
                    Proj_grad = max(Gradient, 0)
                elif 0 < temp_alpha < self.C:
                    Proj_grad = Gradient

                if not np.isclose(Proj_grad, 1e-9):
                    self.alpha[fold_i][ind] = min(max(temp_alpha - Gradient / self.Q_ii[fold_i][ind], 0), self.C)
                    self.beta[fold_i,] = self.beta[fold_i,] + y_train[ind]*(self.alpha[fold_i][ind] - temp_alpha)*X_train[ind,]

    def _check_optimality(self, method, fold_i, X_train, y_train):
        """
        Check optimality condition using duality gap
        params:
            method: str,'duality_gap'; 
                    'gradient': check if all projected gradient are close to 0
                    'gradient_gap': check if the difference between largest and smallest projected gradients are less than a tolerance
        """
        if method == 'duality_gap':
            self._duality_gap(fold_i, X_train, y_train)
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
        self.fold_num = skf.get_n_splits()

        # initialize for dual CD step
        self.Q_ii = []
        self.alpha = []
        self.beta = np.zeros((self.fold_num, feature_size))
        for i, index in enumerate(self.indice_gen):
                
            train_index = index[0]
            test_index = index[1]

            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            self.Q_ii.append((X_train*X_train).sum(axis=1))
            self.alpha.append(np.zeros(X_train.shape[0])) # initialize alpha
            self.beta[i,] = np.dot(np.multiply(self.alpha[i], y_train), X_train)

        t = 1
        # print self.stop()
        # dp = dynamic_plot(xlim=(0,self.t_max), ylim=(0, 1), xlabel = 'Iteration', ylabel = 'Accuracy')
        v = 0 # momentum update
        self.mu = 0.5
        while (self.stop() < self.accuracy_threshold) and (t <= self.t_max):

            C_grad_ls = []
            for i, index in enumerate(self.indice_gen):
                
                train_index = index[0]
                test_index = index[1]

                X_train, X_valid = X[train_index], X[test_index]
                y_train, y_valid = y[train_index], y[test_index]

                #update lower level variables
                self.dual_CD_step(i, X_train, y_train)

                error_vector_t = np.multiply(y_train, np.dot(X_train, self.beta[i,]))
                error_vector_v = np.multiply(y_valid, np.dot(X_valid, self.beta[i,]))

                # l = np.random.randint(X_train.shape[0])
                misclassified_train_ind = np.where(error_vector_t<1)[0]

                temp_grad_C = np.dot(y_train[misclassified_train_ind], X_train[misclassified_train_ind, ])

                p = np.random.choice(np.where(error_vector_v<1)[0])              

                # update upper level gradient
                C_grad_ls.append(-np.dot(y_valid[p]*X_valid[p,], temp_grad_C))
            # print 't', t
            # print C_grad_ls

            C_grad = sum(C_grad_ls)/self.fold_num

            # print C_grad

            # self.lr_C = 1/np.absolute(t*np.sqrt(feature_size)*C_grad)
            # self.C = self.C - self.lr_C*C_grad

            # momentum update
            v = self.mu*v - self.lr_C * C_grad
            self.C += v


            if self.C < self.C_min:
                self.C = self.C_min
            if self.C > self.C_max:
                self.C = self.C_max

            # print self.stop()
            # print self.C
            # dp.update_line(t, self.stop())
            self.accuracy_ls.append(self.stop())
            self.C_ls.append(self.C)
            self.loss_ls.append(self.loss_upper())

            t += 1
        # print 'final accuracy: ', self.stop()
        # dp.fig.savefig('pima_error_profile_stepC_1_stepW_0.001_one_validation_Alg2.png')

        print 'final C: ', self.C
        print 'final cross-val accuracy: ', self.stop()

        f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(18,8))
        ax1.plot(range(0, self.t_max), self.accuracy_ls)
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim((0.5,1))
        ax1.set_xlabel('Iteration')

        ax2.plot(range(0, self.t_max), self.loss_ls)
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Iteration')

        ax3.plot(self.C_ls, self.loss_ls,'*-')
        ax3.set_ylabel('Loss')
        ax3.set_xlabel('C')
        f.suptitle('Magic04 data, Algorithm 2')
        plt.savefig("Figures/Alg2_magic04_mu_%2.1f_alpha_%5.4f_step_decay.png" % (self.mu, self.lr_C), dpi=1000)
        # plt.show()

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

    def loss_upper(self):
        loss = []
        for i, index in enumerate(self.indice_gen):
            test_index = index[1]
            X_valid =  self.X[test_index]
            y_valid =  self.y[test_index]
            loss.append(np.sum( np.maximum(1 - np.multiply(np.dot(X_valid, self.beta[i,]), y_valid), 0)))
        return sum(loss)/self.fold_num


    def _duality_gap(self, fold_i, X_train, y_train):
        dual_obj = -0.5* np.dot(self.beta[fold_i,], self.beta[fold_i,]) + np.sum(self.alpha[fold_i])

        prim_obj = 0.5* np.dot(self.beta[fold_i,], self.beta[fold_i,]) + self.C * np.sum( np.maximum(1 - np.multiply(np.dot(X_train, self.beta[fold_i,]), y_train), 0))

            # print (prim_obj - dual_obj)
        self.gap = prim_obj - dual_obj

    def predict(self, X):

        pred = np.sign(np.dot(X, self.beta))
        return pred

    def accuracy(self, y_true, y_pred):

        return sum(y_true==y_pred)*1.0/len(y_true)


if __name__ == '__main__':
    # X = pd.read_csv('../OptimizationProject_Wei/adult_x.csv', header=None)
    # y = pd.read_csv('../OptimizationProject_Wei/adult_y.csv', header=None)
    X, y = magic04()


    skf = StratifiedKFold(n_splits=5)
    
    bi_SGD = bilevel_SGD_Alg2_5folds()
    bi_SGD.fit(X, y, skf)
