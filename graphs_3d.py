import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn import preprocessing

filename = 'ecoli.csv'
dataset = pd.read_csv(filename, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])

def plot_combo(cols, title):
    comb = combinations(cols, 3)
    combos = []
    for i in list(comb):
        combos.append([i[0], i[1], i[2]])
    combos = np.array(combos)
    grid_side = np.ceil(np.sqrt(len(combos)))
    plot = 0
    fig = plt.figure(title)
    for c in combos:
        X1 = np.asarray([[row[c[0]]] for index, row in dataset.iterrows()])
        X2 = np.asarray([[row[c[1]]] for index, row in dataset.iterrows()])
        X3 = np.asarray([[row[c[2]]] for index, row in dataset.iterrows()])
        Y = np.asarray([[row[8]] for index, row in dataset.iterrows()]).ravel()
        le = preprocessing.LabelEncoder()
        le.fit(Y)
        Y_classes = le.classes_
        Y_cols = Y
        Y_cols[Y_cols == Y_classes[0]] = 'r'
        Y_cols[Y_cols == Y_classes[1]] = 'g'
        Y_cols[Y_cols == Y_classes[2]] = 'b'
        plot += 1
        ax = fig.add_subplot(grid_side, grid_side, plot, projection='3d')
        ax.scatter(X1, X2, X3, c=Y_cols, s=30, alpha=1.0, edgecolor='k')
        ax.set_xlabel('X1 ~ ' + str(c[0]))
        ax.set_ylabel('X2 ~ ' + str(c[1]))
        ax.set_zlabel('X3 ~ ' + str(c[2]))
        ax.set_title('Scatter Plots of ' + filename + ' ' + str(c[0]) + str(c[1]) + str(c[2]))
        ax.grid(True)
    fig.suptitle(title, fontsize=20)
    plt.subplots_adjust(left=0.005, bottom=0.030, right=0.970, top=0.930, wspace=0.200, hspace=0.400)
    return None


plot_combo([1, 2, 3, 4, 5, 6, 7], 'Scatter 3D All Combinations')
plot_combo([1, 2, 5, 6, 7], 'Scatter 3D Selected Combinations')
plot_combo([2, 6, 7], 'Scatter 3D Best Combination')


# End of File