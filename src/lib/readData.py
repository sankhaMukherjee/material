import pandas as pd 
import numpy  as np
import json, os

from tqdm import tqdm

config = json.load(open('../config/config.json'))

def readMaterialProps():

    data = {}

    elements = ['Al', 'In', 'Ga', 'O']

    # first read in all the data
    # --------------------------
    propFolder = config['dataSources']['materialProperties']
    files = os.listdir(propFolder)
    for f in files:
        if 'prop' in f:
            df = pd.read_csv(os.path.join(propFolder, f))
            data['prop'] = df
        else:
            with open(os.path.join(propFolder, f)) as fl:
                lines = [ l.strip().split(',') for l in fl]
            lines = [l for l in lines if l[0] in elements]
            df = pd.DataFrame(lines, columns=['atom', f[:-4]])
            
            data[ f[:-4] ] = df

    # Merge the data into a single table
    # -----------------------------------
    factors = [f[:-4] for f in files if 'prop' not in f]

    for f in factors:
        if 'electronegativity' in f: continue
        data['prop'] = data['prop'].merge(data[f], on='atom')


    return data['prop']

def readGeometryData(fileName):

    latticeProp = []
    atoms = []

    with open(fileName) as f:
        for l in f:

            if l.startswith('#'): 
                continue
            if l.startswith('lattice_vector'):
                latticeProp.append(np.array(list(map(float, l.split()[1:]))))
            if l.startswith('atom'):
                _, x, y, z, a = l.split()
                atoms.append((a, np.array(list(map(float, [x, y, z])))))

    return latticeProp, atoms

def readMaterialDf(trainData=True):

    folder = '../data/intermediate/materialProps'
    files  = os.listdir(folder)
    if trainData:
        files = [f for f in files if 'train' in f]
    else:
        files = [f for f in files if 'test' in f]

    data = []
    for f in tqdm(files):
        data.append(pd.read_csv(os.path.join(folder, f)))
    data = pd.concat(data, axis=1)

    return data

def readSampleData(trainData = True):

    if trainData:
        file = config['dataSources']['train']
    else:
        file = config['dataSources']['test']

    dfX = pd.read_csv(file)
    dfX.drop( 'id', axis=1 )
    yCols = ['formation_energy_ev_natom', 'bandgap_energy_ev' ]
    if list(dfX.columns)[-1] == 'bandgap_energy_ev':
        dfY = dfX[ yCols ]
        dfX = dfX.drop(yCols, axis=1)
    else:
        dfY = None

    return dfX, dfY
