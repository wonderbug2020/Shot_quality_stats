'''
This program will merge my two current best working models into one location
This should steamline the testing process as the models are pretty well set
'''
import pandas as pd
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, r2_score
import matplotlib.pyplot as plt


def get_X_y(df):
    ''' This function takes in a dataframe and splits it into the X and y variables
    '''
    X = df.drop(['is_goal','Unnamed: 0'], axis=1)
    y = df.is_goal

    return X,y

def get_X_y_tab(df):
    ''' This function takes in a dataframe and splits it into the X and y variables
    '''
    X = df.drop(['is_goal'], axis=1).to_numpy()
    y = df.is_goal.to_numpy()

    return X,y

def split_data_tab(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=101)
    y_train = y_train.reshape(-1,1)
    y_val = y_val.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    return X_train, X_val, X_test, y_train, y_val, y_test

def run_XGB(df,test_size=0.25,random_state=101):
    X,y = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    xgb_reg = xgb.XGBRegressor(objective='binary:logistic',
                           eval_metric = 'logloss',
                           eta = .068,
                           subsample = .78,
                           colsample_bytree = .76,
                           min_child_weight = 9,
                           max_delta_step = 5,
                           nthread = 4)
    print('Running XGB model, please be patient')
    xgb_reg.fit(X_train, y_train)
    print('Fitting the test data to the model')
    y_pred = xgb_reg.predict(X_test)
    ll = round(log_loss(y_test, y_pred),5)
    print(f'The Log loss is {ll}')
    r2 = round(r2_score(y_test, y_pred),4)
    print(f'The R^2 value is {r2}')
    return xgb_reg

def run_tabnet(df,max_epochs = 20, device='cpu'):
    X,y = get_X_y_tab(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_tab(X,y)
    reg = TabNetRegressor(device_name = device)
    print('Running the TabNet DNN, this could take a while')
    model = reg.fit(X_train, y_train,
                    eval_set = [(X_train, y_train), (X_val, y_val)],
                    eval_name=['train','val'],
                    eval_metric = ['logloss'],
                    max_epochs = max_epochs)
    print('Fitting the test data to the model')
    y_pred = reg.predict(X_test)
    ll = round(log_loss(y_test, y_pred),5)
    print(f'The Log loss is {ll}')
    return model

def plot_xgb(model, show_feat=10, type='weight'):
    ''' This function uses XGB's plot importance function
        max_num_features is the number of features that you want displayed
        importance_type is what is used to calcualte importance. Options are weight, gain, and cover
    '''
    xgb.plot_importance(model, max_num_features=show_feat, importance_type=type, xlabel=type)
