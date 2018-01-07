import os, json
import numpy             as np
import matplotlib.pyplot as plt

from lib                     import scorer
from sklearn.model_selection import learning_curve

def plotLearningCurve(estimators, X, y, params=None, cv = 5, trainSizes=np.linspace(0.1, 1, 5), n_jobs=-1, fileName='testLC.png'):


    plt.figure(figsize=(7, 3))
    ax1 = plt.axes([0.1, 0.2, 0.39, 0.79])
    ax2 = plt.axes([0.6, 0.2, 0.39, 0.79])

    # first curve
    trainSizes, trainScores, testScores = learning_curve(
        estimators[0], X, y[:, 0], cv=cv, n_jobs=n_jobs, train_sizes=trainSizes,
        scoring = scorer.rmsle_scorer)

    ax1.fill_between(trainSizes, 
        trainScores.mean(axis=1) - trainScores.std(axis=1),
        trainScores.mean(axis=1) + trainScores.std(axis=1),
                     alpha=0.1, color="r")
    ax1.fill_between(trainSizes, 
        testScores.mean(axis=1) - testScores.std(axis=1),
        testScores.mean(axis=1) + testScores.std(axis=1),
                     alpha=0.1, color="g")
    ax1.plot(trainSizes, trainScores.mean(axis=1), 'o-', color="r",
             label="train")
    ax1.plot(trainSizes, testScores.mean(axis=1), 'o-', color="g",
             label="test")
    ax1.set_xlabel('training samples')
    ax1.set_ylabel('formation energy (eV)')

    # Second Curve 
    trainSizes, trainScores, testScores = learning_curve(
        estimators[1], X, y[:, 1], cv=cv, n_jobs=n_jobs, train_sizes=trainSizes,
        scoring = scorer.rmsle_scorer)
    
    ax2.fill_between(trainSizes, 
        trainScores.mean(axis=1) - trainScores.std(axis=1),
        trainScores.mean(axis=1) + trainScores.std(axis=1),
                     alpha=0.1, color="r")
    ax2.fill_between(trainSizes, 
        testScores.mean(axis=1) - testScores.std(axis=1),
        testScores.mean(axis=1) + testScores.std(axis=1),
                     alpha=0.1, color="g")
    ax2.plot(trainSizes, trainScores.mean(axis=1), 'o-', color="r",
             label="train")
    ax2.plot(trainSizes, testScores.mean(axis=1), 'o-', color="g",
             label="test")
    ax2.set_xlabel('training samples')

    ax2.set_ylabel('bandgap (eV)')

    config = json.load(open('../config/config.json'))
    imgFolder = config['results']['imgFolder']
    fileName  = os.path.join( imgFolder, fileName )

    plt.savefig(fileName)

    return
