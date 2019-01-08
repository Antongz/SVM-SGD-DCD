# -*- coding: utf-8 -*-
# read in the data examples used in 
# Couellan N, Wang W. Bi-level stochastic gradient for large scale support vector machine. Neurocomputing 2015;153:300–8. doi:10.1016/J.NEUCOM.2014.11.025
# Author: Wei Jiang
# Date: 1/26/2018
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file


def dummy_categorical(features_final, cat_feature):
    """
    Impute missing values for continuous features and create dummy variable

    :param features_final: all features dataframe before creating dummy variables for categorical features
    :return: dataframe
    """
    for temp_feature in cat_feature:

        just_dummies = pd.get_dummies(features_final[[temp_feature]])
        just_dummies.drop(just_dummies.columns[0],axis =1, inplace =True)
        features_final = pd.concat([features_final, just_dummies], axis=1)
        features_final.drop(temp_feature, inplace=True, axis=1)

    return features_final

def chf():
    # df = pd.read_csv("C:/Users/wjian/Dropbox/HFReadmission/all_features_for_final_logit_with_label.csv")
    df = pd.read_csv("C:/Users/wjian/Dropbox/HFReadmission/selected_final_day_dummy_imputed_with_labels.csv")

    del df['EXTERNAL_ID']
    X = df.values[:, :-1]

    y = df['Readmitted_allcause_30days_flag'].values
    X = X[:, np.var(X, axis=0)>0]

    X = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x))*2 -1, axis=0 , arr = X)
    y[y==0] = -1

    return X, y

def xero():
    """
    Todd's acute xerostomia data
    """
    df = pd.read_csv("data/xerostomia_dose_and_target.csv")
    del df['patientID']
    X = df.values[:, :-1]

    y = df['change'].values

    X = X[:, np.var(X, axis=0)>0]

    X = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x))*2 -1, axis=0 , arr = X)
    y[y==0] = -1

    return X, y

def xero_recovery():
    """
    Todd's xerostomia recovery data
    """
    df = pd.read_csv("data/xerostomia_recovery_dose_data.csv")
    del df['patientID']
    X = df.values[:, :-1]
    y = df['change'].values

    X = X[:, np.var(X, axis=0)>0]

    X = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x))*2 -1, axis=0 , arr = X)
    y[y==0] = -1

    return X, y

def SPECT():
    # df = pd.read_csv("data/SPECT.txt",header=None)
    df = pd.read_csv("data/SPECT_test.txt",header=None)
    df = df.drop_duplicates()

    df = df.sample(frac=1)
    X = df.values[:,1:]
    X = X.astype(np.float64) 

    y = df.values[:,0]
    y[y==0]=-1

    X = X[:, np.var(X, axis=0)>0]

    X = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x))*2 -1, axis=0 , arr = X)

    return X,y 

def connect():
    df = pd.read_csv("data/connect-4.data/connect-4.data", header=None)
    df = df.iloc[df.values[:,-1]!='draw']
    # print np.unique(df.iloc[:,-1])
    y = df.iloc[:,-1].values
    df.iloc[:,-1] = df.iloc[:,-1].map({'win':1, 'loss':-1})
    # y[y=='win']==1
    # y[y=='loss']==-1
    y = df.iloc[:,-1].values
    df=df.iloc[:,0:-1]

    X = dummy_categorical(df, range(df.shape[1]) ).values

    return X, y



def magic04():
    df = pd.read_csv("data/magic04.txt", header=None)

    X = df.values[:,range(0,10)]
    y = df.values[:,10]

    X = X[:, np.var(X, axis=0)>0]

    X = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x))*2 -1, axis=0 , arr = X)

    y[y=='g'] = -1
    y[y=='h'] = 1
    return X, y

def cancer_data():
    df = pd.read_csv("data/breast-cancer-wisconsin.txt", na_values ='?', header=None)
    df = df.dropna(axis='index')

    X = df.values[:,range(1,10)]
    y = df.values[:,10]

    X = X[:, np.var(X, axis=0)>0]

    X = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x))*2 -1, axis=0 , arr = X)
    # X = np.apply_along_axis(lambda x: (x-np.mean(x))/np.std(x), axis=0 , arr = X)


    # print np.apply_along_axis(lambda x: np.mean(x), axis=0 , arr = X)
    y[y==2] = -1
    y[y==4] = 1
    return X, y

def pima_data():
    df = pd.read_csv("data/pima-indians-diabetes.txt")

    X = df.values[:,range(0,8)]
    y = df.values[:,8]

    X = X[:, np.var(X, axis=0)>0]

    X = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x))*2 -1, axis=0 , arr = X)

    y[y==0] = -1
    # y[y==1] = 1
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



def real_sim():
    X, y = load_svmlight_file("data/real-sim/real-sim")
    np.random.seed(0)

    temp_ind = np.random.randint(X.shape[0], size=X.shape[0]//200)
    val_ind = list(set(range(0, X.shape[0])) - set(temp_ind))
    X_train = X[temp_ind,]
    y_train = y[temp_ind,]


    # print X_train[1,:]
    X = X_train.toarray()

    X = X[:, np.var(X, axis=0)>0]

    X = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x))*2 -1, axis=0 , arr = X)
    # y_train = y_train
    # print type(y_train)

    return X, y_train

if __name__ == '__main__':
    X, y = xero_recovery()
    print(X.shape)
    # print X.shape
    # X, y = connect()
    print(sum(y==-1), sum(y==1))
    