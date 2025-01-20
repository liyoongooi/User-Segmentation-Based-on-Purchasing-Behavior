import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from kneed import KneeLocator

def kmeans(data):
    sse = {}
    for k in range(2, 11):
        kmeans = KMeans(n_clusters = k, random_state = 0)
        kmeans.fit(data)
        sse[k] = kmeans.inertia_
    
    x = list(sse.keys())
    y = list(sse.values())
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    k = kn.knee

    return KMeans(n_clusters = k, random_state = 0).fit_predict(data)