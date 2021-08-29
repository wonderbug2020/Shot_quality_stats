'''
This program will merge my two current best working models into one location
This should steamline the testing process as the models are pretty well set
'''
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


def get_X_y(df):
    ''' This function takes in a dataframe and splits it into the X and y variables
    '''
    X = df.drop(['is_goal'], axis=1)
    y = df.is_goal

    return X,y

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
