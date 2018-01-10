import numpy  as np
import pandas as pd
from tqdm import tqdm

def totMaterialProp( N, pctAl, pctGa, pctIn, propAl, propGa, propIn, propO ):

    nAl, nGa, nIn = int((2/5)*N)*np.array([pctAl, pctGa, pctIn])
    nO            = int((3/5)*N)

    totProp = np.array([nAl, nGa, nIn, nO])*np.array([propAl, propGa, propIn, propO])

    return totProp

def newCols(dfX, props, prop='atomic_radii', dataset='train'):

    vals = []
    for r in tqdm(dfX.iterrows(), total=len(dfX)):
        N, pctAl, pctGa, pctIn = r[1].loc[[
            'number_of_total_atoms', 'percent_atom_al', 
            'percent_atom_ga', 'percent_atom_in']].values
        val = totMaterialProp(N, pctAl, pctGa, pctIn, *props['atomic_radii'].values)
        vals.append(val)

    vals = pd.DataFrame(vals, columns=[(prop+'_'+c) for c in ['Al', 'Ga', 'In', 'O']  ])

    vals.to_csv('../data/intermediate/materialProps/{}_{}_total.csv'.format(prop, dataset), index=False)

    return vals
