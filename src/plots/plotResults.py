import os, json
import numpy             as np
import matplotlib.pyplot as plt

config = json.load(open('../config/config.json'))


def plotHist(y, yHat, fileName):

    fileName = fileName.replace('.png', '_Hist.png')

    plt.figure(figsize=(7, 3))
    ax1 = plt.axes([0.1, 0.2, 0.39, 0.79])
    ax2 = plt.axes([0.6, 0.2, 0.39, 0.79])

    y = np.log(y + 1)
    yHat = np.log(yHat + 1)

    score0 = np.sqrt(((y[:, 0] - yHat[:, 0])**2).mean())
    score1 = np.sqrt(((y[:, 1] - yHat[:, 1])**2).mean())

    ax1.hist((yHat[:, 0] - y[:, 0])**2, bins=100, facecolor='brown', alpha=0.6)
    ax2.hist((yHat[:, 1] - y[:, 1])**2, bins=100, facecolor='brown', alpha=0.6)
    
    ax1.set_xlabel('formation energy (eV) - {}'.format(score0))
    ax1.set_ylabel('log squre err')
    ax2.set_xlabel('bandgap (eV) - {}'.format(score1))
    ax2.set_ylabel('log squre err')

    plt.savefig(fileName)
    plt.close()

    return


def plotLogErrors(y, yHat, fileName, colors=None):

    fileName = fileName.replace('.png', '_Log-errors.png')

    plt.figure(figsize=(7, 3))
    ax1 = plt.axes([0.1, 0.2, 0.39, 0.79])
    ax2 = plt.axes([0.6, 0.2, 0.39, 0.79])

    y = np.log(y + 1)
    yHat = np.log(yHat + 1)

    if colors is None:
        ax1.plot(y[:, 0], (yHat[:, 0] - y[:, 0])**2, 's', mfc='orange', mec='brown', alpha=0.1)
    else:
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(vmin=colors.min(), vmax=colors.max()))
        sm._A = []
        plt.colorbar(sm)
        c = ax1.scatter(y[:, 0], (yHat[:, 0] - y[:, 0])**2, marker='s', c=colors, alpha=0.1)
    ax1.axhline(0, lw=2, color='black')

    if colors is None:
        ax2.plot(y[:, 1], (yHat[:, 1]-y[:, 1])**2, 's', mfc='orange', mec='brown', alpha=0.1)
    else:
        ax2.scatter(y[:, 1], (yHat[:, 1] - y[:, 1])**2, marker='s', c=colors, alpha=0.1)
    ax2.axhline(0, lw=2, color='black')

    
    ax1.set_xlabel('formation energy (eV)')
    ax1.set_ylabel('log square error')
    ax2.set_xlabel('bandgap (eV)')
    ax2.set_ylabel('log square error')

    plt.savefig(fileName)
    plt.close()

    return

def plotErrors(y, yHat, fileName, colors=None):

    fileName = fileName.replace('.png', '_errors.png')

    plt.figure(figsize=(7, 3))
    ax1 = plt.axes([0.1, 0.2, 0.39, 0.79])
    ax2 = plt.axes([0.6, 0.2, 0.39, 0.79])

    if colors is None:
        ax1.plot(y[:, 0], yHat[:, 0] - y[:, 0], 's', mfc='orange', mec='brown', alpha=0.1)
    else:
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(vmin=colors.min(), vmax=colors.max()))
        sm._A = []
        plt.colorbar(sm)
        c = ax1.scatter(y[:, 0], yHat[:, 0] - y[:, 0], marker='s', c=colors, alpha=0.1)

    ax1.axhline(0, lw=2, color='black')

    if colors is None:
        ax2.plot(y[:, 1], yHat[:, 1]-y[:, 1], 's', mfc='orange', mec='brown', alpha=0.1)
    else:
        ax2.scatter(y[:, 1], yHat[:, 1] - y[:, 1], marker='s', c=colors, alpha=0.1)
        
    ax2.axhline(0, lw=2, color='black')
    

    ax1.set_xlabel('formation energy (eV)')
    ax1.set_ylabel('error')
    ax2.set_xlabel('bandgap (eV)')
    ax2.set_ylabel('error')


    plt.savefig(fileName)
    plt.close()

    return

def plotPredictions(y, yHat, fileName, colors=None):

    imgFolder = config['results']['imgFolder']
    fileName  = os.path.join( imgFolder, fileName )

    plt.figure(figsize=(7, 3))
    ax1 = plt.axes([0.1, 0.2, 0.39, 0.79])
    ax2 = plt.axes([0.6, 0.2, 0.39, 0.79])

    if colors is not None:
        ax1.scatter(y[:, 0], yHat[:, 0], marker='s', c=colors, alpha=0.1)
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(vmin=colors.min(), vmax=colors.max()))
        sm._A = []
        plt.colorbar(sm)
    else:
        ax1.plot(y[:, 0], yHat[:, 0], 's', mfc='orange', mec='brown', alpha=0.1)
    ax1.plot([y[:, 0].min(), y[:, 0].max()], [y[:, 0].min(), y[:, 0].max()], lw=2, color='black' )

    if colors is not None:
        ax2.scatter(y[:, 1], yHat[:, 1], marker='s', c=colors, alpha=0.1)
    else:
        ax2.plot(y[:, 1], yHat[:, 1], 's', mfc='orange', mec='brown', alpha=0.1)
    ax2.plot([y[:, 1].min(), y[:, 1].max()], [y[:, 1].min(), y[:, 1].max()], lw=2, color='black' )


    ax1.set_xlabel('formation energy (eV)')
    ax1.set_ylabel('prediction')
    ax2.set_xlabel('bandgap (eV)')
    ax2.set_ylabel('prediction')


    plt.savefig(fileName)
    plt.close()

    plotErrors(y, yHat, fileName, colors)
    plotLogErrors(y, yHat, fileName, colors)
    plotHist(y, yHat, fileName)

    return