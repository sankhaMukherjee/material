from lib                     import scorer
from models                  import GBmodel
from sklearn.model_selection import RandomizedSearchCV
from datetime                import datetime as dt

import json, os
import pandas            as pd
# import seaborn           as sns
import matplotlib.pyplot as plt 

config = json.load(open('../config/config.json'))

def getParams(paramDist, temp):

    cols = list(paramDist.keys())
    paramList = [dict(r) for r in temp[:, cols].iterrows()]
    print(paramList)

    return paramList

def optimizeModel(estimator, paramDist, X, y, modelType='RF', name='bandGap', verbose=True):

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')

    print('Starting randomized grid search ...')

    rsParams = json.load(open('../config/RandomizedGridSearchCVParams.json'))
    search  = RandomizedSearchCV(estimator, paramDist, scoring = scorer.rmsle_scorer ,**rsParams)
    search.fit(X, y)
    results = search.cv_results_.copy()

    print('Finishing randomized grid search ...')
    print('Result:')

    temp = results['params']
    for i, t in enumerate(temp):
        t['rank_test_score'] = results['rank_test_score'][i]
        t['mean_test_score'] = results['mean_test_score'][i]
        t['std_test_score'] = results['std_test_score'][i]

    temp = pd.DataFrame(temp).sort_values('rank_test_score')
    temp.reset_index(drop=True, inplace=True)

    newCols = [
        'rank_test_score',
        'mean_test_score',
        'std_test_score']

    columns = [c for c in list(temp.columns) if c not in newCols]
    columns = newCols + columns
    temp = temp[columns]

    # Make sure that you save the results 
    # ------------------------------------
    file = '../results/optResults/optParams-{}-{}-{}.csv'.format(
                modelType, name, now)
    temp.to_csv(file, index=False)

    return temp

def getLatestOptimizers(name='GB'):

    folder  = '../results/optResults'

    filesBG = [ f for f in os.listdir(folder) if (name in f) and ('bandGap' in f)]
    filesBG = sorted(filesBG, reverse=True)
    BGparams = pd.read_csv(os.path.join(folder, filesBG[0]))

    filesFE = [ f for f in os.listdir(folder) if (name in f) and ('formationEnergy' in f)]
    filesFE = sorted(filesFE, reverse=True)
    FEparams = pd.read_csv(os.path.join(folder, filesFE[0]))

    return BGparams, FEparams


