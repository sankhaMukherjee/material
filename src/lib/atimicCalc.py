import numpy as np

def findVolume(a, b, c, alpha, beta, gamma):
    '''returns the volume given the lattice parameters
    
    For the description of the volume parameters, look here:
    https://en.wikipedia.org/wiki/Lattice_constant#Volume
    
    Arguments:
        a {[type]} -- [description]
        b {[type]} -- [description]
        c {[type]} -- [description]
        alpha {[type]} -- [description]
        beta {[type]} -- [description]
        gamma {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    '''

    # Obtain the direction cosines
    angles  = np.radians([alpha, beta, gamma])
    cosines = np.cos(angles)
    volume  = np.sqrt(1 + 2*cosines.prod() - (cosines**2).sum())

    return volume

def findDistanceMatrix(atoms):



    return