import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from graph_filters import graph_filtering
from utils import suggest, spacesize
from scipy.io import loadmat
from hyperopt import fmin, hp, STATUS_OK, Trials
import numpy as np
from sklearn.metrics import f1_score

warnings.simplefilter('ignore')

runs = 5


def objective(params):
    x = features.copy()
    if method:
        x = graph_filtering(x, lmbda=params['lambda'], nn=params['nn'], degree=params['degree'],
                            alpha=params['alpha'], t=params['t'], method=method)
    model = LogisticRegression(C=params['C'])
    _scaler = StandardScaler()
    x = _scaler.fit_transform(x)
    model.fit(x[idx_train], labels[idx_train])
    y_pred = model.predict(x[idx_val])
    f1 = f1_score(labels[idx_val], y_pred, average='macro')
    return {'loss': -f1, 'status': STATUS_OK}


for method in [
    None,  # no filter
    'sgc',
    's2gc',
    'dgc',
    'appnp'
]:
    print(f'{method}:')
    for dataset in [
        'classic4',
        'dbpedia',
        'ohsumed',
        'R8',
        '20ng',
        'ag_news',
        'bbc',
        'classic3'
    ]:
        print(f'  {dataset}:')

        data = loadmat(f'data/embeddings/{dataset}-embedding.mat')
        features = data['x']
        labels = data['y'].reshape(-1)

        n_classes = len(np.unique(labels))

        splits = loadmat(f'data/splits/{dataset}-split.mat')
        idx_train, idx_val, idx_test = splits['train'].reshape(-1), splits['val'].reshape(-1), splits['test'].reshape(
            -1)

        degrees = [1, 5, 10, 50]
        lambdas = [1, 10]
        nn = [3, 5, 10]
        alphas = [.1, .2]
        t = [4, 6]
        C = [.1, 1]

        space = {
            'degree': hp.choice('degree', degrees),
            'alpha': hp.choice('alpha', alphas if method == 'appnp' else [1]),
            'lambda': hp.choice('lambda', lambdas),
            'nn': hp.choice('nn', nn),
            't': hp.choice('t', t if method == 'dgc' else [1]),
            'C': hp.choice('C', C)
        }

        # hyperparameter tuning
        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    trials=trials,
                    algo=suggest,
                    max_evals=spacesize(space))

        if method:
            features = graph_filtering(features,
                                       lmbda=lambdas[best['lambda']],
                                       nn=nn[best['nn']],
                                       degree=degrees[best['degree']],
                                       alpha=alphas[best['alpha']], t=t[best['t']],
                                       method=method
                                       )
        # Transductive classification
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        metrics = {'f1': [], 'time': []}

        for run in range(runs):
            lr = LogisticRegression(C=C[best['C']])
            lr.fit(features[idx_train], labels[idx_train])
            Z = lr.predict(features[idx_test])

            metrics['f1'].append(f1_score(labels[idx_test], Z, average='macro'))

        results = {
            'mean': {k: (np.mean(v)).round(4) for k, v in metrics.items()},
            'std': {k: (np.std(v)).round(4) for k, v in metrics.items()}
        }
        means = results['mean']
        stds = results['std']

        print(f'    f1:  {means["f1"]}')
