import os, csv, json
import numpy as np 
from datetime import datetime as dt

config = json.load(open('../config/config.json'))

def saveData(yHat, models=None):

    folder   = config['results']['uploadFolder']
    fileName = dt.now().strftime('%Y-%m-%d--%H-%M-%S.csv')
    fileName = os.path.join(folder, fileName)

    with open(fileName, 'w') as f:
        f.write('id,formation_energy_ev_natom,bandgap_energy_ev\n')
        w = csv.writer(f)
        for i, r in enumerate(yHat):
            r0, r1 = r
            if r0 < 0: 
                r0 = 0
            if r1 < 0: 
                r1 = 0
            w.writerow( [i+1, r0, r1] )

    return
