import json
import numpy as np

from tqdm             import tqdm
from sklearn          import model_selection as MS
from sklearn.metrics  import accuracy_score
from sklearn.ensemble import RandomForestRegressor


def solveRegressor(rfConfig, X, y, saveLoc=None, initModel=None, CV=False):
    '''Generate and run an RF model
    
    This function is used for generating an RF mdoel, and then running
    it. If an initial model is provided, then this will load the 
    model provided, and then use that model as an initializer. The 
    model generated will then be given a hot start from the initial 
    model. 
    
    Arguments:
        rfConfig {dict} -- the dictionary of hyperparameters
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
        initModel {RandomForestRegressor() model} -- This is the result 
            of an earlier fitted model. In case one is provided, the 
            current model will be restarted from this model (default: {None}
            in which case, a new model will be generated.)
    
    Returns:
        RandomForestRegressor() -- This is the result of a fitted model,
            given the data and the rest of the parameters. 
    '''


    if initModel is None:
        rfModel = RandomForestRegressor(**rfConfig)
    
    if CV:

        # We want to make sure that the information is meaningful
        # for all splits. Otherwise, its pretty meaningless ...
        # Obtain hyperparameters from the JSON file. This obviously
        # Takes a long time. So, we shall use this for testing only
        # -----------------------------------------------------------
        rkfFact = json.load(open('../config/RepeatedKFold.json'))
        rkf     = MS.RepeatedKFold(**rkfFact)

        scores = []
        for train_index, test_index in tqdm(rkf.split(X), total=rkfFact['n_splits']*rkfFact['n_repeats'] ):

            # We want to make sure that we start with
            # the provided model in every split. Otherwsie
            # we will be training on top of the other models
            # as warm-start is 1. 
            if initModel is not None:
                rfModel = initModel
                rfModel.set_parameter(warm_start=True)

            rfModel.fit(X[train_index, :], y[train_index])
            yHat  = rfModel.predict(X[test_index])
            score = 0
            score = np.sqrt( ((yHat - y[test_index])**2).mean() )
            
            scores.append(score)
            tqdm.write('Score = ({}) {}'.format( np.mean(scores), score ))

        # Refitting the model with the whole data
        if initModel is not None:
            rfModel = initModel
            rfModel.set_parameter(warm_start=True)

        print('Score summary: {} +-({})'.format( np.mean(scores), np.std(scores) ))
        print('Percentage difference: {}'.format( 100*np.mean(scores)/np.mean(y) ))

    rfModel.fit( X, y )

    return rfModel

