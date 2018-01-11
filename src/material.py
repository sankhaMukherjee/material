import json
import numpy as np 
import pandas as pd
from tqdm import tqdm

from pprint import pprint
from lib    import readData, saveData, scorer, addMaterialProp
from numpy  import linalg as LA
from models import rfmodel, GBmodel, optimizeModel, XGboost
from plots  import plotResults, plotLearningCurves
from tqdm   import tqdm


def main():

    props = readData.readMaterialProps()
    # latticeProp, atoms = readData.readGeometryData('../data/raw_data/test/1/geometry.xyz')
    
    # atoms1 = np.array([a[1] for a in atoms])
    # print(LA.norm(atoms1 - atoms1[0], axis=1))

    dfX, dfy   = readData.readSampleData(trainData = True)
    dfX1, dfy1 = readData.readSampleData(trainData = False)

    # Plot data about the input ...
    if True:
        print('X ...')
        print(dfX.head())
        print('\nMaterial Props')
        print(props)

        propsCols = list(props.columns)
        propsCols = [c for c in propsCols if c not in ['atom']]

        for prop in tqdm(propsCols):
            # tqdm.write('\n{}'.format(prop))
            # tqdm.write(str(props[prop].values))
            addMaterialProp.newCols(dfX, props, prop=prop, dataset='train')
            addMaterialProp.newCols(dfX1, props, prop=prop, dataset='test')
        
        # print(pd.concat((dfX, vals), axis=1))

    if False:

        i = 0
        # Get optimized parameters for each model
        params = optimizeModel.getLatestOptimizers(name='XGboost')
        print(params[0].head())
        print(params[1].head())

        for i in tqdm(range(1)):

            # Initiate the first model
            xgbConfig = json.load(open('../config/XGBconfig.json'))
            m1 = XGboost.solveRegressor( 
                xgbConfig, dfX.values, dfy.values[:, 0].flatten(), saveLoc=None)

            temp = dict(params[0].iloc[i, 3:])
            temp['max_depth']    = int(temp['max_depth'])
            temp['n_estimators'] = int(temp['n_estimators'])
            for k in temp:
                xgbConfig[k] = temp[k]

            m1.set_params(**xgbConfig)
            m1 = XGboost.solveRegressor( 
                xgbConfig, dfX.values, dfy.values[:, 0].flatten(), saveLoc=None)

            # Do the same for the second model
            xgbConfig = json.load(open('../config/XGBconfig.json'))
            m2 = XGboost.solveRegressor( 
                xgbConfig, dfX.values, dfy.values[:, 1].flatten(), saveLoc=None)

            temp = dict(params[1].iloc[i, 3:])
            temp['max_depth']    = int(temp['max_depth'])
            temp['n_estimators'] = int(temp['n_estimators'])
            for k in temp:
                xgbConfig[k] = temp[k]
            
            m2.set_params(**xgbConfig)
            m2 = XGboost.solveRegressor( 
                xgbConfig, dfX.values, dfy.values[:, 1].flatten(), saveLoc=None)

            yHat = np.hstack((
                m1.predict(dfX.values).reshape(-1, 1), 
                m2.predict(dfX.values).reshape(-1, 1)))

            plotResults.plotPredictions(dfy.values, yHat, 'XGBoptimized_{:05d}.png'.format(i), dfX.ix[:, 'spacegroup'].values)

            # Predict the result using this method ...
            yHat = np.hstack((
                m1.predict(dfX1.values).reshape(-1, 1), 
                m2.predict(dfX1.values).reshape(-1, 1)))

            saveData.saveData(yHat)

    if False:
        # Optimize a XGboost model and save the result ...        
        xgbConfig = json.load(open('../config/XGBconfig.json'))
        m1 = XGboost.solveRegressor( 
            xgbConfig, dfX.values, dfy.values[:, 0].flatten(), saveLoc=None)
        temp = optimizeModel.optimizeModel(m1, 
            XGboost.optParams, dfX.values, dfy.values[:, 0].flatten(), 
            modelType='XGboost', name='formationEnergy')

        m2 = XGboost.solveRegressor( 
            xgbConfig, dfX.values, dfy.values[:, 1].flatten(), saveLoc=None)
        temp = optimizeModel.optimizeModel(m2, 
            XGboost.optParams, dfX.values, dfy.values[:, 1].flatten(), 
            modelType='XGboost', name='bandGap')


    if False:
        xgbConfig = json.load(open('../config/XGBconfig.json'))

        m1 = XGboost.solveRegressor( 
            xgbConfig, dfX.values, dfy.values[:, 0].flatten(), saveLoc=None)
        m2 = XGboost.solveRegressor( 
            xgbConfig, dfX.values, dfy.values[:, 1].flatten(), saveLoc=None)

        yHat = np.hstack((
            m1.predict(dfX.values).reshape(-1, 1), 
            m2.predict(dfX.values).reshape(-1, 1)))

        plotResults.plotPredictions(dfy.values, yHat, 'XGBmodel.png')

        # yHat = np.hstack((
        #     m1.predict(dfX1.values).reshape(-1, 1), 
        #     m2.predict(dfX1.values).reshape(-1, 1)))

        # saveData.saveData(yHat)

    if False:

        i = 0
        # Get optimized parameters for each model
        params = optimizeModel.getLatestOptimizers(name='GB')
        print(params[0].head())
        print(params[1].head())

        for i in range(1):

            # Initiate the first model
            gbConfig = json.load(open('../config/GradientBoostingRegressor.json'))
            m1 = GBmodel.solveRegressor( 
                gbConfig, dfX.values, dfy.values[:, 0].flatten(), saveLoc=None, initModel=None, CV=False)

            temp = dict(params[0].iloc[i, 3:])
            for k in temp:
                gbConfig[k] = temp[k]

            m1.set_params(**gbConfig)
            m1 = GBmodel.solveRegressor( 
                gbConfig, dfX.values, dfy.values[:, 0].flatten(), saveLoc=None, initModel=m1, CV=False)

            # Do the same for the second model
            gbConfig = json.load(open('../config/GradientBoostingRegressor.json'))
            m2 = GBmodel.solveRegressor( 
                gbConfig, dfX.values, dfy.values[:, 1].flatten(), saveLoc=None, initModel=None, CV=False)

            temp = dict(params[1].iloc[i, 3:])
            for k in temp:
                gbConfig[k] = temp[k]
            
            m2.set_params(**gbConfig)
            m2 = GBmodel.solveRegressor( 
                gbConfig, dfX.values, dfy.values[:, 1].flatten(), saveLoc=None, initModel=m2, CV=False)

            yHat = np.hstack((
                m1.predict(dfX.values).reshape(-1, 1), 
                m2.predict(dfX.values).reshape(-1, 1)))

            plotResults.plotPredictions(dfy.values, yHat, 'GBsmallOptimization_{:05d}.png'.format(i))

            # Predict the result using this method ...
            yHat = np.hstack((
                m1.predict(dfX1.values).reshape(-1, 1), 
                m2.predict(dfX1.values).reshape(-1, 1)))

            saveData.saveData(yHat)

    if False:
        
        # Optimize a model and save the result ...        
        gbConfig = json.load(open('../config/GradientBoostingRegressor.json'))
        m1 = GBmodel.solveRegressor( 
            gbConfig, dfX.values, dfy.values[:, 0].flatten(), saveLoc=None, initModel=None, CV=False)
        temp = optimizeModel.optimizeModel(m1, 
            GBmodel.optParams, dfX.values, dfy.values[:, 0].flatten(), 
            modelType='GB', name='formationEnergy')

        m2 = GBmodel.solveRegressor( 
            gbConfig, dfX.values, dfy.values[:, 1].flatten(), saveLoc=None, initModel=None, CV=False)
        temp = optimizeModel.optimizeModel(m2, 
            GBmodel.optParams, dfX.values, dfy.values[:, 1].flatten(), 
            modelType='GB', name='bandGap')
        
    if False:
        gbConfig = json.load(open('../config/GradientBoostingRegressor.json'))
        m1 = GBmodel.solveRegressor( 
            gbConfig, dfX.values, dfy.values[:, 0].flatten(), saveLoc=None, initModel=None, CV=False)
        m2 = GBmodel.solveRegressor( 
            gbConfig, dfX.values, dfy.values[:, 1].flatten(), saveLoc=None, initModel=None, CV=False)
        plotLearningCurves.plotLearningCurve([m1, m2], dfX.values, dfy.values, fileName='GBlearn.png')

    if False:
        gbConfig = json.load(open('../config/GradientBoostingRegressor.json'))

        m1 = GBmodel.solveRegressor( 
            gbConfig, dfX.values, dfy.values[:, 0].flatten(), saveLoc=None, initModel=None, CV=False)
        m2 = GBmodel.solveRegressor( 
            gbConfig, dfX.values, dfy.values[:, 1].flatten(), saveLoc=None, initModel=None, CV=False)

        yHat = np.hstack((
            m1.predict(dfX.values).reshape(-1, 1), 
            m2.predict(dfX.values).reshape(-1, 1)))

        plotResults.plotPredictions(dfy.values, yHat, 'GradientBoostModel_again.png')

        yHat = np.hstack((
            m1.predict(dfX1.values).reshape(-1, 1), 
            m2.predict(dfX1.values).reshape(-1, 1)))

        # saveData.saveData(yHat)

    if False:
        rfConfig = json.load(open('../config/RandomForestRegressor.json'))

        m1 = rfmodel.solveRegressor( 
            rfConfig, dfX.values, dfy.values[:, 0].flatten(), saveLoc=None, initModel=None, CV=False)
        m2 = rfmodel.solveRegressor( 
            rfConfig, dfX.values, dfy.values[:, 1].flatten(), saveLoc=None, initModel=None, CV=False)

        yHat = np.hstack((
            m1.predict(dfX.values).reshape(-1, 1), 
            m2.predict(dfX.values).reshape(-1, 1)))

        plotResults.plotPredictions(dfy.values, yHat, 'RFmodel.png')

        yHat = np.hstack((
            m1.predict(dfX1.values).reshape(-1, 1), 
            m2.predict(dfX1.values).reshape(-1, 1)))

        saveData.saveData(yHat)

    return

if __name__ == '__main__':
    main()