# -*- coding: utf-8 -*-
# This script implements the modified (my algorithm 1) bilevel SGD method for SVM hyperparameter tuning in this paper:
# Couellan N, Wang W. Bi-level stochastic gradient for large scale support vector machine. Neurocomputing 2015;153:300–8. doi:10.1016/J.NEUCOM.2014.11.025
# Author: Wei Jiang
# Date: 3/19/2018
import numpy as np
import random as rd
import pandas as pd
from dynamic_plot import dynamic_plot
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import StratifiedKFold
from data_example import *
matplotlib.rcParams.update({'font.size': 22})
from bilevel_SGD_5_folds import bilevel_SGD
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class bilevel_SGD_Alg1_5folds():

    def __init__(self):
        self.C = None
        self.alpha = None
        self.beta = None
        self.C_min = 1e-4
        self.C_max = 1e6
        self.t_max = 200 # maximal number of iterations
        self.lr_beta = 0.001 # learning rate (step size) for beta
        self.lr_C = 0.001 # learning rate for C

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
                # ind = np.random.randint(X_train.shape[0])
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
                else:
                    print('temp_alpha: ', temp_alpha, 'C: ', self.C)



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
        X = np.c_[ X, np.ones(X.shape[0]) ] # add a constant column for intercept
        self.X = X
        self.y = y

        train_ind_ls = []
        test_ind_ls = []
        for train_ind, test_ind in skf.split(X, y):
            train_ind_ls.append(train_ind)
            test_ind_ls.append(test_ind)
        self.indice_gen = list(zip(train_ind_ls, test_ind_ls))
        # print(self.indice_gen)

        self.C = self.C_min
        feature_size = self.X.shape[1]
        self.fold_num = skf.get_n_splits()

        # initialize for dual CD step
        self.M_ii = []
        self.Q_ii = []
        self.alpha = []
        self.beta = np.zeros((self.fold_num, feature_size))
        for i, index in enumerate(self.indice_gen):
                
            train_index = index[0]
            test_index = index[1]

            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]

            self.M_ii.append(np.multiply(y_train, np.transpose(X_train)))
            self.Q_ii.append((X_train*X_train).sum(axis=1))
            self.alpha.append(np.zeros(X_train.shape[0])) # initialize alpha
            self.beta[i,] = np.dot(np.multiply(self.alpha[i], y_train), X_train)

        t = 1
        v = 0 # momentum update
        self.mu = 0.5
        t2 = 1

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

                #update lower level variables
                self.dual_CD_step(i, X_train, y_train)

                error_vector_t = np.multiply(y_train, np.dot(X_train, self.beta[i,]))
                error_vector_v = np.multiply(y_valid, np.dot(X_valid, self.beta[i,]))

                J_alpha = (self.alpha[i] >= self.C)

                if any(J_alpha):
                    p = np.random.choice(np.where(error_vector_v<1)[0])
                    # ind_p = np.where(error_vector_v<1)[0]

                    # update upper level gradient
                    # C_grad = - np.dot(y_valid[p]*X_valid[p,], np.dot(self.M, J_alpha))
                    C_grad_ls.append(-np.dot(np.dot(y_valid[p], X_valid[p,]), np.dot(self.M_ii[i], J_alpha)))

            if C_grad_ls:
                C_grad = sum(C_grad_ls)/self.fold_num
                # print C_grad

                # print C_grad
                if C_grad !=0:
                    self.lr_C = 1/np.absolute(t*np.sqrt(feature_size)*C_grad)
                # self.lr_c = 1.0/t
                # self.lr_C = 1/np.absolute(t*np.sqrt(feature_size))

                    self.C = self.C - self.lr_C*C_grad

                # # momentum update
                # v = self.mu*v - self.lr_C * C_grad
                # self.C += v


                if self.C < self.C_min:
                    self.C = self.C_min
                if self.C > self.C_max:
                    self.C = self.C_max

                # t2 += 1

            # print self.stop()
            # print self.C
            # dp.update_line(t, self.stop())
            

            t += 1
        print('final C: ', self.C)
        print('final cross-val accuracy: ', self.stop())
        # print 't2: ', t2

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
            # print('sum(accuracy_v_ls)', sum(accuracy_v_ls))
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

def plot_result(t, accuracy_ls_1, loss_ls_1, C_ls_1, accuracy_ls_2, loss_ls_2, C_ls_2, title):
    '''
    1 represents results for our approach, Alg.2, 2 represents result for bilevel_SGD
    '''

    # f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,10))
    f, (ax1, ax2) = plt.subplots(1,2,figsize=(14,8))

    ax1.plot(range(0, t-1), accuracy_ls_1, 'b', range(0, t-1), accuracy_ls_2, 'k')
    # ax1.set_ylim(0.5,1)
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Iteration')

    # ax2.plot(range(0, t-1), loss_ls_1, 'b', range(0, t-1), loss_ls_2, 'k')
    l3, = ax2.plot(range(0, t-1), loss_ls_1, 'b', label='SGD+DCD')
    l4, = ax2.plot(range(0, t-1), loss_ls_2, 'k', label = 'Bilevel-SGD')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Iteration')

    # l3, = ax3.plot(C_ls_1, loss_ls_1,'b*-', label='SGD+DCD')
    # l4, = ax3.plot(C_ls_2, loss_ls_2,'k*-', label = 'Bilevel-SGD')
    # ax3.set_ylabel('Loss')
    # ax3.set_xlabel('C')
    f.suptitle(title)
    plt.legend( handles=[l3, l4],loc="lower center",
           ncol=1,  fancybox=True, fontsize=16,  bbox_to_anchor=(0.8,0.99))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.85,
                wspace=0.25, hspace=None)
    # plt.savefig('C:/Users/wjian/Dropbox/phd_research/svm_hyper_parameter_optimization/Code_Gradient_Method/Algorithm1-2/'+title+'.png',dpi=1000)
    plt.show()

    plt.figure(figsize=(10,8))
    # plt.plot(range(0, t-1), C_ls_1, 'b*-', range(0, t-1), C_ls_2, 'k*-', markersize=4)

    DCD, = plt.plot(range(0, t-1), C_ls_1, 'b*-', label= 'SGD+DCD', markersize=4)
    SGD, = plt.plot(range(0, t-1), C_ls_2, 'k*-', label= 'Bilevel-SGD', markersize=4)
    plt.legend( handles=[DCD, SGD], loc="lower center",
           ncol=1,  fancybox=True, fontsize=16,  bbox_to_anchor=(0.86,0.99))
    plt.xlabel('Iteration')
    plt.ylabel('C')
    plt.title(title)
    # plt.savefig('C:/Users/wjian/Dropbox/phd_research/svm_hyper_parameter_optimization/Code_Gradient_Method/Algorithm1-2/C_profile_'+title+'.png',dpi=1000)
    plt.show()


if __name__ == '__main__':
    # X = pd.read_csv('../OptimizationProject_Wei/adult_x.csv', header=None)
    # y = pd.read_csv('../OptimizationProject_Wei/adult_y.csv', header=None)
    X, y = pima_data()
    print(X.shape)
    np.random.seed(1)

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    print('Bilevel SGD:')
    SGD = bilevel_SGD()
    SGD.t_max = 150
    tm2 = time.time()
    t2,accuracy_ls_2, loss_ls_2, C_ls_2 = SGD.fit(X, y, skf)
    # print C_ls_2
    print(len(accuracy_ls_2), 'length')
    tm3 = time.time()
    print('running time:', tm3-tm2)

    print('\n SGD+DCD:')
    bi_SGD = bilevel_SGD_Alg1_5folds()
    bi_SGD.t_max = 150
    tm0 = time.time()
    t1,accuracy_ls_1, loss_ls_1, C_ls_1 = bi_SGD.fit(X, y, skf)
    print(len(accuracy_ls_1), 'length')
    tm1 = time.time()
    print('running time:', tm1-tm0)

    parameters = { 'C':[1e-4, 0.1, 1, 10, 100, 1e6, bi_SGD.C, SGD.C]}
    svc = LinearSVC(random_state=0, loss='hinge', max_iter=150, tol=1e-5, fit_intercept=True)
    clf = GridSearchCV(svc, parameters, cv=5, scoring='accuracy', refit='False')
    tm0 = time.time()
    clf.fit(X, y)
    tm1 = time.time()

    print('\n sklearn GridSearchCV:')
    print('best CV score:', clf.best_score_)
    print('best hyperparameter:', clf.best_params_)
    print('running time:', tm1-tm0)
    print(clf.cv_results_['mean_test_score'])

    plot_result(t1, accuracy_ls_1, loss_ls_1, C_ls_1, accuracy_ls_2, loss_ls_2, C_ls_2, 'Real Sim Data')
