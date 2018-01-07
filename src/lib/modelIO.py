from sklearn.externals import joblib
import os, json
from datetime import datetime as dt

def saveModel(model, modelType='RF', predicts='bandGap', otherInfo={}):

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
    folder = '../models/{}-{}-{}'.format(modelType, predicts, now)
    os.mkdirs(folder)

    with open(os.path.join(folder, 'params.json'), 'w') as f:
        params = model.get_params()
        json.dump(params, f)

    with open(os.path.join(folder, 'otherInfo.json'), 'w') as f:
        json.dump(otherInfo, f)

    joblib.dump(model, os.path.join(folder, 'model.pkl'))
    
    return
