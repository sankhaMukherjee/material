import json
import numpy   as np

from tqdm             import tqdm
from sklearn          import model_selection as MS
from xgboost          import XGBRegressor
from sklearn.metrics  import accuracy_score

optParams = {
    'max_depth'         : [3, 5, 7, 10, 15, 20],
    'learning_rate'     : [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5], 
    'n_estimators'      : [100, 500, 1000, 2000], 
    # 'silent'            : True, 
    # 'objective'         : 'reg:linear', 
    'booster'           : ['gbtree', 'gblinear', 'dart'], 
    # 'n_jobs'            : -1, 
    # 'nthread'           : None, 
    # 'gamma'             : 0, 
    # 'min_child_weight'  : 1, 
    # 'max_delta_step'    : 0, 
    # 'subsample'         : 1, 
    # 'colsample_bytree'  : 1, 
    # 'colsample_bylevel' : 1, 
    'reg_alpha'         : [0, 0.1, 0.3, 0.7, 0.9, 1], 
    'reg_lambda'        : [0, 0.1, 0.3, 0.7, 0.9, 1], 
    # 'scale_pos_weight'  : 1, 
    'base_score'        : [0.1, 0.3, 0.6, 1, 2, 3], 
    # 'random_state'      : 0, 
    # 'seed'              : 2018, 
    # 'missing'           : None,
}

def solveRegressor(xgbConfig, X, y, saveLoc=None):
    '''Generate and run an GB model
    
    This function is used for generating an GB mdoel, and then running
    it. If an initial model is provided, then this will load the 
    model provided, and then use that model as an initializer. The 
    model generated will then be given a hot start from the initial 
    model. 
    
    Arguments:
        xgbConfig {dict} -- the dictionary of hyperparameters
        X {numpy 2d array} -- The array of values that will
            be used for generating a prediction. 
        y {numpy 1d array} -- The expected result that we want
            the model to train to. 
    
    Keyword Arguments:
        saveLoc {str} -- Location where the model should be saved. 
            This assumes that the location whre the model is to be
            saved will be writable and exists. Remember, at this time
            the function just does not do any error checking. (default: 
            {None}, in which case, the model is not saved. )

    Returns:
        GradientBoostingRegressor() -- This is the result of a fitted model,
            given the data and the rest of the parameters. 
    '''


    xgbModel = XGBRegressor(**xgbConfig)
    xgbModel.fit( X, y )

    return xgbModel


