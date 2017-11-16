import math
import random
import numpy as np
from itertools import combinations
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn import preprocessing


#normalize the data
def normalize_KM(a):
    b = np.apply_along_axis(lambda x: (x - np.mean(x)), 0, a)
    return b

#normalize the data
def normalize_GM(a):
    b = np.apply_along_axis(lambda x: (x-np.mean(x)),0,a)
    col_ind = []
    for col in range(b.shape[1]):
        le = sum(b[:,col])
        if le == 0:
            col_ind.append(col)
    b = np.delete(b,np.s_[col_ind],axis =1)
    return b

#load the file and normalize the data
def load_csv(file, col1, col2, col3):
    dataset = pd.read_csv(file, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    X = np.asarray([[row[col1], row[col2], row[col3]] for index, row in dataset.iterrows()])
    Y = np.asarray([[row[8]] for index, row in dataset.iterrows()]).ravel()
    X = X.astype(np.float)
    #X = normalize(X) # Not Performing Normalization
    return X, Y

#cluster assignment for each point
def cluster_assignment_KM(x, y, initial_centroids):
    cluster_dict = {}
    labelsDict = {}
    for i, row in enumerate(x):
        smallest_dist = 9999
        cluster_id = -1
        label = y[i]
        for k, centroid in enumerate(initial_centroids):
            dist = np.sqrt(sum((centroid - row) ** 2))
            if dist < smallest_dist:
                smallest_dist = dist
                cluster_id = k
        if cluster_id in cluster_dict:
            l = cluster_dict[cluster_id]
            l = np.vstack([l, row])
            cluster_dict[cluster_id] = l
            labelsDict[cluster_id].append(label)
        else:
            cluster_dict[cluster_id] = row
            labelsDict[cluster_id] = [label]
    return cluster_dict, labelsDict

#cluster assignment for each point
def cluster_assignment_GM(x,initial_centroids):
    cluster_dict = {}
    for row in x:
        smallest_dist = 9999
        cluster_id = -1
        for k,centroid in enumerate(initial_centroids):
            dist = np.sqrt(sum((centroid - row) ** 2))
            if dist < smallest_dist:
                smallest_dist = dist
                cluster_id = k
        if cluster_id in cluster_dict:
            l = cluster_dict[cluster_id]
            l = np.vstack([l, row])
            cluster_dict[cluster_id] = l
        else:
            cluster_dict[cluster_id] = row
    return cluster_dict

#return the centroid value for each cluster
def get_centroids_KM(cluster_dict):
    centroid_dict = {}
    for k in cluster_dict.keys():
        l = cluster_dict[k]
        li = np.mean(l, axis=0)
        centroid_dict[k] = li
    return centroid_dict

#calculate sum squared error
def cal_sse_KM(centroid_dict, cluster_dict):
    s = 0
    for k in cluster_dict.keys():
        x = cluster_dict[k]
        centroid = centroid_dict[k]
        for i in range(x.shape[0]):
            dist = np.sqrt(sum((centroid - x[i]) ** 2))
            s += dist
    return s

#calculate sum squared error 
def cal_sse_GM(x,label_dict,centroids):
    sse = 0
    for i in range(x.shape[0]):
        label_val = label_dict[i]
        centroid_val = centroids[label_val]
        dist = np.sqrt(sum((centroid_val - x[i]) ** 2))
        sse += dist
    return sse

# Implementing the Multivariate Gaussian Density Function
def Gausssian_normal_GM(x, mu, cov):
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = np.exp(-.5 * np.einsum('ij, ij -> i',x - mu, np.dot(np.linalg.inv(cov),(x - mu).T).T ) )
    return part1 * np.exp(part2)

#Gaussian cluster estimation
def cluster_estimation_GM(R,x,centroids,covar,pi):
    for k in range(R.shape[1]):
        tmp = pi[k] * Gausssian_normal_GM(x,centroids[k], covar[k])
        R[:,k] = tmp
    # Normalize the responsibility matrix
    R = (R.T / np.sum(R, axis = 1)).T
    return R

def getLabelsGroupDict_KM(labels):
    labelsDict = {}
    for i in labels:
        labelsDict[i] = labelsDict.get(i, 0) + 1
    return labelsDict

def getLabelsGroupDict_GM(labels):
    labelsDict = {}
    for i in labels:
        labelsDict[i] = labelsDict.get(i,0)+1
    return labelsDict

#get the entropy of the class labels
def entropyOfClassLabels_KM(labels):
    totalLabels = len(labels)
    labelsDict = getLabelsGroupDict_KM(labels)
    hy = 0
    for key, val in labelsDict.items():
        t = float(val / float(totalLabels))
        hy += t * (math.log(t, 2)) * (-1.0)
    return hy

#Calculation of entropy of class labels
def entropyOfClassLabels_GM(labels):
    totalLabels = len(labels)
    labelsDict = getLabelsGroupDict_GM(labels) # Get the no of instances belonging to a particular class
    hy = 0
    for key,val in labelsDict.items():
        t = float(val/float(totalLabels))
        hy += t*(math.log(t,2))*(-1.0)
    return hy

def cal_hyc_each_cluster_GM(cluster_values,y):
    class_dict = {}
    len_of_cluster = len(cluster_values)
    #build a dict with class_values and count
    for x in cluster_values:
        class_dict[y[x]] = class_dict.get(y[x], 0) + 1
    #cal the hyc value now
    hyc_cluster = 0
    for key in class_dict.keys():
        val = float(class_dict[key])/len_of_cluster
        hyc_cluster += val*(math.log(val,2))*(-1.0)
    return hyc_cluster

#NMI value
def getNMIValue_KM(clusterLabelDict, hy):
    # Calculate entropy of cluster labels
    #hcList = []
    labelsSumList = [len(l) for l in clusterLabelDict.values()]
    totalLabels = sum(labelsSumList)
    #Calculating entropy of cluster labels here itself to improve performance rather
    #than calling entropyOfClassLabels method
    #Iterate to get the total number of labels
    hc = 0
    for val in labelsSumList:
        #totalLabels += len(val)
        #hcList.extend([key for k in val])
        t = float(val / float(totalLabels))
        hc += t * (math.log(t, 2)) * (-1.0)
    #hc = entropyOfClassLabels(hcList)
    hyc = 0
    for key, val in clusterLabelDict.items():
        p = float(len(val) / float(totalLabels))
        hyc += p * entropyOfClassLabels_KM(val)
    # calculate normalized mutual information
    nmi = float(2.0 * (hy - hyc) / (hy + hc))
    return nmi

def getNMIValue_GM(clusterDict,hy,y):
    #Calculate entropy of cluster labels
    hc = 0
    total_no_of_instances = len(clusterDict.values())
    for val in clusterDict.keys():
        t = float(len(clusterDict[val]))/total_no_of_instances
        hc += t*(math.log(t,2))*(-1)
    #calculate the hyc
    hyc = 0
    for k in clusterDict.keys():
        x_of_cur_cluster = clusterDict[k]
        hyc_cluster = cal_hyc_each_cluster_GM(x_of_cur_cluster,y)
        hyc_cluster = float(len(x_of_cur_cluster)/total_no_of_instances)* hyc_cluster
        hyc += hyc_cluster
    #calculate NMI now - normalized mutual information
    nmi = float(2.0*(hy - hyc)/(hy+hc))
    return nmi

def getOptimalK(cost_func_list, kvals):
    diffs = []
    for i in range(int(len(cost_func_list) - 1)):
        diffs.append(cost_func_list[i] - cost_func_list[i + 1])
    maxcostdiff = np.max(diffs)
    maxcostindex = np.where(diffs == maxcostdiff)
    maxcostindex = maxcostindex[0][0]
    costfuncprev = cost_func_list[maxcostindex]
    costfuncnext = cost_func_list[maxcostindex + 1]
    if (costfuncprev > costfuncnext):
        kindex = np.where(cost_func_list == costfuncnext)
        kindex = kindex[0][0]
    elif (costfuncprev < costfuncnext):
        kindex = np.where(cost_func_list == costfuncprev)
        kindex = kindex[0][0]
    else:
        pass
    return kindex


def do_KMeans(col1, col2, col3):
    filename = 'ecoli.csv'
    no_clusters = 4
    # filename and columns
    X, Y = load_csv(filename, col1, col2, col3)
    hy = entropyOfClassLabels_KM(Y)
    cost_func_list = []
    bestClusters = []
    finalCentroids = []
    for i in range(1, no_clusters):
        best_sse = 9999
        bestClusterLabelDict = {}
        for j in range(75):
            x = X
            y = Y
            centroids = np.empty(shape=[0, X.shape[1]])
            # initialize centroids to random points
            for k in range(i + 1):
                ran = random.randint(1, x.shape[0])
                centroids = np.vstack([centroids, x[ran, :]])
                #x = np.delete(x, ran, axis=0)
                #y = np.delete(y, ran, axis=0)
            #k-means algorithm
            cluster_dict = {}
            centroid_dict = {}
            max_ite = 50
            ite = 0
            labelsDict = {}
            while ite < max_ite:
                #cluster assignment step
                cluster_dict, labelsDict = cluster_assignment_KM(x, y, centroids)
                #move centroids step
                centroid_dict = get_centroids_KM(cluster_dict)
                centroids = centroid_dict.values()
                #print (centroids)
                ite += 1
            sse = cal_sse_KM(centroid_dict, cluster_dict)
            sse = sse / float(X.shape[0])
            if sse < best_sse:
                best_sse = sse
                bestClusterLabelDict = labelsDict
                bestClusterDict = cluster_dict
        nmi = getNMIValue_KM(bestClusterLabelDict, hy)
        print ("~" * 50)
        print ("Number Of Clusters : ", (i + 1))
        print ("SSE Value : ", best_sse)
        print ("NMI Value : ", nmi)
        print ("Final Centroids : " + str(centroid_dict))
        finalCentroids.append([(i + 1), centroid_dict])
        bestClusters.append([(i + 1), bestClusterLabelDict])
        for key, val in bestClusterLabelDict.items():
            print ("ClusterID : ", key, " >>> Labels in Cluster : ", getLabelsGroupDict_KM(val))
        print ("~" * 50)
        cost_func_list.append(best_sse)
    print ("CostFunc List : ", cost_func_list)
    kvals = [k for k in range(2, (no_clusters + 1))]
    print ("K Values : ", kvals)
    cost_func_list = np.array(cost_func_list)
    optKindex = getOptimalK(cost_func_list, kvals)
    optK = kvals[optKindex]
    print ("~" * 75)
    print ("Optimal K value : " + str(optK))
    optCentroids = finalCentroids[optKindex][1]
    optCentroids_list = []
    for key, val in optCentroids.items():
        print ("Centroid " + str(key) + " : " + str(val))
        optCentroids_list.append(val)
    optCentroids_list = np.array(optCentroids_list).reshape(3, 3)
    #print ("Centroids List : " + str(optCentroids_list))
    optCluster = bestClusters[optKindex][1]
    optYlabels = []
    for key, val in optCluster.items():
        print ("ClusterID : ", key, " >>> Labels in Cluster : ", getLabelsGroupDict_KM(val))
        print (optCluster[key])
        optYlabels = optYlabels + optCluster[key]
    print ("Optimal Cluster Labels : " + str(optYlabels))
    print ("~" * 75)
    plt.figure("K Means Algorithm Elbow Plot " + str(col1) + str(col2) + str(col3))
    plt.suptitle("K Means Algorithm Elbow Plot " + str(col1) + str(col2) + str(col3))
    plt.plot(range(2, no_clusters + 1), cost_func_list)
    plt.plot(kvals[optKindex], cost_func_list[optKindex], 'ro')
    plt.ylabel("SSE")
    plt.xlabel("K Value ")
    plt.grid(True)
    
    # Plot the Original Clusters
    fig = plt.figure("Clusters ~ Original Classes " + str(col1) + str(col2) + str(col3))
    Yo = Y.ravel()
    le = preprocessing.LabelEncoder()
    le.fit(Yo)
    Y_classes = le.classes_
    Y_cols = Yo
    Y_cols[Y_cols == Y_classes[0]] = 'r'
    Y_cols[Y_cols == Y_classes[1]] = 'g'
    Y_cols[Y_cols == Y_classes[2]] = 'b'
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y_cols, s=30, alpha=1.0, edgecolor='k')
    ax.set_xlabel('X1 ~ ' + str(col1))
    ax.set_ylabel('X2 ~ ' + str(col2))
    ax.set_zlabel('X3 ~ ' + str(col3))
    ax.set_title('Scatter Plots of ' + filename + ' ' + str(col1) + str(col2) + str(col3))
    ax.grid(True)
    fig.suptitle("Clusters ~ Original Classes", fontsize=20)
    plt.subplots_adjust(left=0.005, bottom=0.030, right=0.970, top=0.930, wspace=0.200, hspace=0.400)
    
    # Plot the K-Means Clusters
    fig = plt.figure("Clusters ~ K-Means Classes " + str(col1) + str(col2) + str(col3))
    Yc = np.asarray(optYlabels).ravel()
    C1 = optCentroids_list[:, 0]
    C2 = optCentroids_list[:, 1]
    C3 = optCentroids_list[:, 2]
    le = preprocessing.LabelEncoder()
    le.fit(Yc)
    Y_classes = le.classes_
    Y_cols = Yc
    Y_cols[Y_cols == Y_classes[0]] = 'r'
    Y_cols[Y_cols == Y_classes[1]] = 'g'
    Y_cols[Y_cols == Y_classes[2]] = 'b'
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y_cols, s=30, alpha=1.0, edgecolor='k')
    ax.scatter(C1, C2, C3, c=['m', 'm', 'm'], s=100, alpha=1.0, edgecolor='k', marker='*')
    ax.set_xlabel('X1 ~ ' + str(col1))
    ax.set_ylabel('X2 ~ ' + str(col2))
    ax.set_zlabel('X3 ~ ' + str(col3))
    ax.set_title('Scatter Plots of ' + filename + ' ' + str(col1) + str(col2) + str(col3))
    ax.grid(True)
    fig.suptitle("Clusters ~ K-Means Classes", fontsize=20)
    plt.subplots_adjust(left=0.005, bottom=0.030, right=0.970, top=0.930, wspace=0.200, hspace=0.400)
    
    return optCentroids_list


def do_GMM(col1, col2, col3, init_centroid):
    #col1, col2, col3 = 2, 6, 7
    filename = 'ecoli.csv'
    no_clusters = 4
    # filename and columns
    X, Y = load_csv(filename, col1, col2, col3)
    hy = entropyOfClassLabels_GM(Y)
    #print X
    log_likelihoods = []
    cost_func_list = []
    finalCentroids = []
    bestClusters = []
    nnmi_list = []
    for i in range(1, (no_clusters + 1)):
        best_sse = 999999
        best_cluster_dict = {}
        for j in range(100):
            x = X
            #centroids = np.empty(shape=[0, X.shape[1]])
            centroids = np.array(init_centroid)
            #initialize centroids to random points
            for k in range(i+1):
                ran=random.randint(1, x.shape[0])
                centroids = np.vstack([centroids, x[ran,:]])
                #x = np.delete(x,ran, axis=0)
            #initial covariance matrix
            covar = [np.cov(x.T)] * i
            # initialize the probabilities/weights for each gaussians
            pi = [1./i] * i
            # responsibility matrix is initialized to all zeros
            R = np.zeros((x.shape[0], i))
            cluster_dict = {}
            max_ite = 50
            ite = 0
            while ite < max_ite:
                #cluster assignment step
                R = cluster_estimation_GM(R, x, centroids, covar, pi)
                # The number of data points belonging to each gaussian
                no_data_points = np.sum(R, axis = 0)
                #update means, covariance, pi
                for k in range(i):
                    # means
                    centroids[k] = 1. / no_data_points[k] * np.sum(R[:, k] * x.T, axis = 1).T
                    x_mu = np.matrix(x - centroids[k])
                    ## covariances
                    covar[k] = np.array(1 / no_data_points[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
                    ## and finally the probabilities
                    pi[k] = 1. / x.shape[0] * no_data_points[k]
                # Likelihood computation
                log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
                cluster_label_dict = {}
                label_dict = {}
                for a,val in enumerate(R):
                    l = val
                    max = 0
                    best_cluster_ind = 0
                    for cluster_ind,val in enumerate(l):
                        if val > max:
                            max = val
                            best_cluster_ind = cluster_ind
                    label_dict[a] = best_cluster_ind
                    if best_cluster_ind in cluster_dict:
                        li = cluster_dict[best_cluster_ind]
                        li.append(a)
                        cluster_dict[best_cluster_ind] = li
                    else:
                        l = []
                        l.append(a)
                        cluster_dict[best_cluster_ind] = l
                    cluster_label_dict[best_cluster_ind] = cluster_label_dict.get(best_cluster_ind, 0) + 1
                sse = cal_sse_GM(x, label_dict, centroids)
                sse = sse/float(X.shape[0])
                if sse < best_sse:
                    best_sse = sse
                    best_cluster_dict = cluster_dict
                ite +=1
        nmi = getNMIValue_GM(best_cluster_dict, hy, Y)
        print ("~" * 50)
        print ("Number Of Clusters = ", i+1)
        print ("SSE Value : ", best_sse)
        print ("NMI Value : ", nmi)
        #print ("Final Centroids : " + str(centroids[-1]))
        finalCentroids.append([(i + 1), centroids[-1]])
        bestClusters.append([(i + 1), best_cluster_dict])
        #for key, val in best_cluster_dict.items():
            #print ("ClusterID: ", key, " >>> Labels in Cluster : ", val)
        print ("~" * 50)
        cost_func_list.append(best_sse)
        nnmi_list.append(nmi)
    print ("CostFunc List : ", cost_func_list)
    kvals = [k for k in range(2, (no_clusters + 1))]
    print ("K Values : ", kvals)
    print ("NMI list", nnmi_list)
    cost_func_list = np.array(cost_func_list)
    optKindex = getOptimalK(cost_func_list, kvals)
    optK = kvals[optKindex]
    print ("~" * 75)
    print ("Optimal K value : " + str(optK))
    #optCentroid = finalCentroids[optKindex + 1][1]
    #print ("Centroid : " + str(optCentroid))
    optCluster = bestClusters[optKindex + 1][1]
    lenClusters = []
    for key, val in optCluster.items():
        lenClusters.append([key, len(val)])
    lenClusters = np.array(lenClusters).reshape(3, 2)
    maxlen = np.max(lenClusters[:, 1])
    maxlenindex = np.where(lenClusters[:, 1] == maxlen)
    maxlenindex = maxlenindex[0][0]
    maxlenkey = lenClusters[maxlenindex][0]
    optYlabels = []
    clusters = []
    for key, val in optCluster.items():
        if (key != maxlenkey):
            #print ("ClusterID : ", key, " >>> Labels in Cluster : ", val)
            clusters.append(val)
            optYlabels = optYlabels + optCluster[key]
    labels = [l for l in range(Y.shape[0])]
    leftover_points = []
    for l in labels:
        if (l not in optYlabels):
            leftover_points.append(l)
    #print ("Leftover Points : " + str(leftover_points))
    clusters.append(leftover_points)
    #print ("Optimal Cluster Labels : " + str(optYlabels))
    for i in range(len(clusters)):
        print ("Cluster " + str(i) + " : " + str(clusters[i]))
    print ("~" * 75)
    #plt.figure("GMM Algorithm Elbow Plot " + str(col1) + str(col2) + str(col3))
    #plt.suptitle("GMM Algorithm Elbow Plot " + str(col1) + str(col2) + str(col3))
    #plt.plot(range(2, no_clusters + 1), cost_func_list[0 : -1])
    #plt.plot(kvals[optKindex], cost_func_list[optKindex], 'ro')
    #plt.ylabel("SSE")
    #plt.xlabel("K Value ")
    #plt.grid(True)
    
    # Plot the Original Clusters
    fig = plt.figure("Clusters ~ Original Classes " + str(col1) + str(col2) + str(col3))
    Yo = Y.ravel()
    le = preprocessing.LabelEncoder()
    le.fit(Yo)
    Y_classes = le.classes_
    Y_cols = Yo
    Y_cols[Y_cols == Y_classes[0]] = 'r'
    Y_cols[Y_cols == Y_classes[1]] = 'g'
    Y_cols[Y_cols == Y_classes[2]] = 'b'
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y_cols, s=30, alpha=1.0, edgecolor='k')
    ax.set_xlabel('X1 ~ ' + str(col1))
    ax.set_ylabel('X2 ~ ' + str(col2))
    ax.set_zlabel('X3 ~ ' + str(col3))
    ax.set_title('Scatter Plots of ' + filename + ' ' + str(col1) + str(col2) + str(col3))
    ax.grid(True)
    fig.suptitle("Clusters ~ Original Classes", fontsize=20)
    plt.subplots_adjust(left=0.005, bottom=0.030, right=0.970, top=0.930, wspace=0.200, hspace=0.400)
    
    # Plot the GMM Clusters
    cluster_cols = ['b', 'g', 'r']
    fig = plt.figure("Clusters ~ GMM Classes " + str(col1) + str(col2) + str(col3))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for l in range(len(clusters)):
        for c in clusters[l]:
            ax.scatter(X[:, 0][c], X[:, 1][c], X[:, 2][c], c=cluster_cols[l], s=30, alpha=1.0, edgecolor='k')
    ax.set_xlabel('X1 ~ ' + str(col1))
    ax.set_ylabel('X2 ~ ' + str(col2))
    ax.set_zlabel('X3 ~ ' + str(col3))
    ax.set_title('Scatter Plots of ' + filename + ' ' + str(col1) + str(col2) + str(col3))
    ax.grid(True)
    fig.suptitle("Clusters ~ GMM Classes", fontsize=20)
    plt.subplots_adjust(left=0.005, bottom=0.030, right=0.970, top=0.930, wspace=0.200, hspace=0.400)
    
    return None


# comb = combinations([1, 2, 5, 6, 7], 3)
# comb = combinations([2, 5, 6, 7], 3)
comb = combinations([2, 6, 7], 3)
combos = []
for i in list(comb):
    combos.append([i[0], i[1], i[2]])
combos = np.array(combos)
for c in combos:
    print ("K-Means Clustering Algorithm")
    optCentroids = do_KMeans(c[0], c[1], c[2])
    initial_centroid = [np.mean(optCentroids[:, 0]), 
                        np.mean(optCentroids[:, 1]), 
                        np.mean(optCentroids[:, 2])]
    print ("Gaussian Mixture Model Algorithm")
    do_GMM(c[0], c[1], c[2], initial_centroid)


# End of File