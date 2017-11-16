import itertools

import matplotlib as mpl

import matplotlib.pyplot as plt

from scipy import linalg

import numpy as np

from sklearn import datasets

from sklearn.mixture import GaussianMixture

from sklearn.model_selection import StratifiedKFold

from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import pairwise_distances_argmin

import pandas as pd


colors = ['navy', 'turquoise', 'darkorange']

X = pd.read_csv("ecoli_pruned_3k.csv", names=[

    '1', '2', '3', '4'])

X = X[['1', '2', '3']]

k_means = KMeans(n_clusters=3)

k_means.fit(X)

labels = k_means.labels_


gmm = GaussianMixture(

    n_components=5, covariance_type='full', init_params='kmeans')

gmm.fit(X)


fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
X = X.values
ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(3), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)

ax.set_xticks(())
ax.set_yticks(())

plt.show()

for n, color in enumerate(colors):
    plt.scatter(X[:, 0], gmm.predict(X), s=0.8, color=color,
                label=labels)

plt.show()
