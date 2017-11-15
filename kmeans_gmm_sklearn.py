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
        print(Y_[i])
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        plt.scatter(i, Y_[i], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
        i = i + 1

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


X = pd.read_csv("ecoli.csv", names=['0', '1', '2', '3', '4', '5', '6', '7', '8'])
X = X[['2', '6', '7']]

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