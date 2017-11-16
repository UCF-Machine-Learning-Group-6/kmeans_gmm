import itertools

import matplotlib as mpl

import matplotlib.pyplot as plt

from scipy import linalg

import numpy as np

from sklearn import datasets

from sklearn.mixture import GaussianMixture

from sklearn.model_selection import StratifiedKFold

from sklearn.cluster import KMeans

from sklearn import datasets

import pandas as pd

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',

                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):

    splot = plt.subplot(2, 1, 1 + index)
    i = 0
    for l, (mean, covar, color) in enumerate(zip(

            means, covariances, color_iter)):

        plt.scatter(i, Y_[i], .8, color=color)

        i = i + 1

    plt.xlim(-9., 5.)

    plt.ylim(-3., 6.)

    plt.xticks(())

    plt.yticks(())

    plt.title(title)


X = pd.read_csv("ecoli_pruned_3k.csv", names=['1', '2', '3', '4'])

X = X[['1', '2', '3']]

est = KMeans(n_clusters=3)

est.fit(X)

labels = est.labels_

X['labels'] = labels

gmm = GaussianMixture(

    n_components=5, covariance_type='full', init_params='kmeans')

gmm.fit(X)

X = X.values

plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,

             'Gaussian Mixture')

plt.show()


# End of File