import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, silhouette_score

def dbscan(data):
    param_grid = {
        'eps': np.linspace(0.1, 5.0, 4),
        'min_samples': range(2 * data.shape[1] - 10, 2* data.shape[1] - 7)
    }
    grid_search = GridSearchCV(
        estimator=DBSCAN(),
        param_grid=param_grid,
        scoring=make_scorer(lambda estimator, X: silhouette_score(X, estimator.fit_predict(X))
            if len(set(estimator.fit_predict(X))) > 1 and -1 not in set(estimator.fit_predict(X))
            else -1),
        cv=[(slice(None), slice(None))],
        verbose=1
    )
    grid_search.fit(data)
    best_params = grid_search.best_params_
    
    return DBSCAN(eps = best_params['eps'], min_samples = best_params['min_samples']).fit_predict(data)