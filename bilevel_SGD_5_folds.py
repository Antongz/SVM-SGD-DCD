# -*- coding: utf-8 -*-
# This script implements the bilevel SGD method for SVM hyperparameter tuning in this paper:
# Couellan N, Wang W. Bi-level stochastic gradient for large scale support vector machine. Neurocomputing 2015;153:300–8. doi:10.1016/J.NEUCOM.2014.11.025
# Author: Wei Jiang
# Date: 1/18/2018
import numpy as np
import random as rd
import pandas as pd
from dynamic_plot import dynamic_plot
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from data_example import *


class bilevel_SGD():

    def __init__(self):
        self.C = None
        self.alpha = None
        self.beta = None
        self.C_min = 1e-4
        self.C_max = 1e6
        self.t_max = 200 # maximal number of iterations
        self.lr_beta = 0.001 # learning rate (step size) for beta
        self.lr_C = 0.0005 # learning rate for C

        self.accuracy_threshold = 0.97

        self.C_ls = []
        self.loss_ls = []
        self.accuracy_ls = []

    def fit(self, X, y, skf):
        """
        params:
            X: array, data, each row is a sample
            y: 1d array, prediction label
            skf: object, sklearn stratifiedKfold 
        """

        ## initialize parameters
        X = np.c_[ X, np.ones(X.shape[0]) ] # add a constant column for intercept
        self.X = X
        self.y = y
        train_ind_ls = []
        test_ind_ls = []
        for train_ind, test_ind in skf.split(X, y):
            train_ind_ls.append(train_ind)
            test_ind_ls.append(test_ind)
        self.indice_gen = list(zip(train_ind_ls, test_ind_ls))
        

        self.C = self.C_min
        feature_size = X.shape[1]
        self.fold_num = skf.get_n_splits(self.X, self.y)

        self.beta = np.random.uniform(-1, 1, (self.fold_num, feature_size))

        t = 1
        # print self.stop()
        # dp = dynamic_plot(xlim=(0,self.t_max), ylim=(0, 200), xlabel = 'Iteration', ylabel = 'Loss')
        while (self.stop() < self.accuracy_threshold) and (t <= self.t_max):

            ############# Record loss metric, and C ########################
            self.accuracy_ls.append(self.stop())
            self.C_ls.append(self.C)
            self.loss_ls.append(self.loss_upper())

            C_grad_ls = []
            for i, index in enumerate(self.indice_gen):
                
                train_index = index[0]
                test_index = index[1]

                X_train, X_valid = X[train_index], X[test_index]
                y_train, y_valid = y[train_index], y[test_index]

                # print type(y_train), type(self.beta[i,])
                error_vector_t = np.multiply(y_train, np.dot(X_train, self.beta[i,]))
                error_vector_v = np.multiply(y_valid, np.dot(X_valid, self.beta[i,]))

                # l = np.random.randint(X_train.shape[0])
                # print np.where(error_vector_t<1)[0]
                l = np.random.choice(np.where(error_vector_t<1)[0])
                p = np.random.choice(np.where(error_vector_v<1)[0])

                misclassified_train_ind = np.where(error_vector_t<1)[0]
                temp_grad_C = np.dot(y_train[misclassified_train_ind], X_train[misclassified_train_ind, ])

                # if y_train[l]*np.dot(self.beta[i, ], X_train[l,]) < 1:
                    # G_grad =  self.beta[i,] - self.C*y_train[l]*X_train[l,]
                # else:
                    # G_grad = self.beta[i,]
                # update lower level gradient
                G_grad =  self.beta[i,] - self.C*y_train[l]*X_train[l,]
                self.lr_beta = 1.0/t
                self.beta[i,] = self.beta[i,] - self.lr_beta*G_grad

                # update upper level gradient
                C_grad_ls.append(- np.dot(y_valid[p]*X_valid[p,], y_train[l]*X_train[l,]))
                # C_grad_ls.append(-np.dot(np.dot(y_valid[p,], X_valid[p,]), temp_grad_C))
            # print 't', t
            # print C_grad_ls
            # print C_grad_ls
            C_grad = sum(C_grad_ls)/self.fold_num
            # print C_grad
            # C_grad = np.random.choice(C_grad_ls)
            # print 'C_grad', C_grad

            # if C_grad == 0:
            #     print 'why'
            #     break

            
            
            # print self.beta
            if C_grad != 0:
                self.lr_C = 1/np.absolute(t*np.sqrt(feature_size)*C_grad)
                # self.lr_C = 1/np.absolute(t*np.sqrt(feature_size))


                self.C = self.C - self.lr_C*C_grad

            if self.C < self.C_min:
                self.C = self.C_min
            if self.C > self.C_max:
                self.C = self.C_max

            # print self.stop()
            # print self.C

            # dp.update_line(t, self.loss_upper())


            t += 1
        # dp.fig.savefig('pima_loss_profile_stepC_1_stepW_0.001.png')

        print('final C: ', self.C)
        print('final cross-val accuracy: ', self.stop())
        

        # plt.figure(figsize=(8,6))
        # plt.plot(range(0, t-1), self.loss_ls)
        # plt.ylabel('Loss', fontsize=18)
        # plt.xlabel('Iteration', fontsize=18)
        # plt.show()


        # f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(18,8))
        # ax1.plot(range(0, t-1), self.accuracy_ls)
        # ax1.set_ylabel('Accuracy')
        # ax1.set_xlabel('Iteration')

        # ax2.plot(range(0, t-1), self.loss_ls)
        # ax2.set_ylabel('Loss')
        # ax2.set_xlabel('Iteration')

        # ax3.plot(self.C_ls, self.loss_ls,'*-')
        # ax3.set_ylabel('Loss')
        # ax3.set_xlabel('C')
        # # f.suptitle('Magic04 data, bilevel SGD')
        # plt.show()
        
        return t, self.accuracy_ls, self.loss_ls, self.C_ls


        


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
    # X = pd.read_csv('../OptimizationProject_Wei/adult_x.csv', header=None)
    # y = pd.read_csv('../OptimizationProject_Wei/adult_y.csv', header=None)
    X, y = xero_recovery()

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    
    bi_SGD = bilevel_SGD()
    bi_SGD.fit(X, y, skf)
