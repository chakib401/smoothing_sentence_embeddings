from sklearn.cluster import KMeans
from graph_filters import graph_filtering
from scipy.io import loadmat
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from time import time

runs = 5
degree = 2

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

        n_clusters = len(np.unique(labels))

        t0 = time()
        if method:
            features = graph_filtering(features, method=method)
        pre_time = time() - t0
        metrics = {'ami': [], 'ari': [], 'time': []}

        for run in range(runs):
            t0 = time()
            Z = KMeans(n_clusters, n_init=10).fit_predict(features)

            metrics['time'].append(pre_time + time() - t0)
            metrics['ami'].append(ami(labels, Z))
            metrics['ari'].append(ari(labels, Z))

        results = {
            'mean': {k: (np.mean(v)).round(4) for k, v in metrics.items()},
            'std': {k: (np.std(v)).round(4) for k, v in metrics.items()}
        }
        means = results['mean']
        stds = results['std']

        print(f'\tmeans: ', means['ami'], means['ari'], sep='&')
        print(f'\tstds: ', stds['ami'], stds['ari'], sep='&')

