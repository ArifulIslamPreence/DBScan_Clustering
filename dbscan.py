import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def create_datapoints(centroidLocation, numSamples, clasterDiviation):
    X,y = make_blobs(n_samples= numSamples, centers= centroidLocation, cluster_std= clasterDiviation)
    X = StandardScaler().fit_transform(X)
    return X,y

X,y = create_datapoints([[4,3],[2,-1],[-1,4]],1500, 0.5)

epsilon = 0.3
minimumPoints = 7
db = DBSCAN(eps=epsilon,min_samples=minimumPoints).fit(X)
labels = db.labels_

core_sample_mask = np.zeros_like(db.labels_,dtype=bool)
core_sample_mask[db.core_sample_indices_] = True

n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
unique_label = set(labels)

colors = plt.cm.Spectral(np.linspace(1,0,len(unique_label)))

for k,col in zip(unique_label,colors):
    if k == -1:
        col = 'k'

    class_member_mask = (labels==k)

    xy = X[class_member_mask & core_sample_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

    xy = X[class_member_mask & ~core_sample_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker=u'o', alpha=0.5)

plt.show()