from __future__ import division
import numpy as np
import sys
import random as rd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from scipy.stats import multivariate_normal as MvN

"""
This script should be launched from the console using the following command line:
python script_file X_data Label_data nb_cluster Clustering_type nb_iter
ex:    > python Clustering.py X.csv Y.csv 3 Kmeans 50
or:    > python Clustering.py X.csv Y.csv 3 GMM 50
Where:
.script_file is the name of this python file
.X_data is a .csv file of the data nodes coordinates
there should be one column per dimension, so for instance 2 columns if we are in 2D
.Label_data is a .csv file that should contain the known labels corresponding 
to the nodes in the X_data file
.nb_cluster is the number of clusters we want to partition the data into (5 is the maximum number we can ask for).
.Clustering_type is the type of clustering we want to apply: 
Kmeans if we want to apply hard-clustering
GMM if we want to apply soft-clustering
.nb_iter is the number of iterations we are going to use
"""


class Kmeans:
    def __init__(self, nb_clust, nb_iter):
        self.X = np.genfromtxt(sys.argv[1], delimiter=",")
        self.Y = np.genfromtxt(sys.argv[2], delimiter=",")
        self.nb_clust = nb_clust
        self.nb_iter = nb_iter

        # Random initialisation of the mean mu = (mu_1,...,m_K)
        self.mu = np.zeros((nb_clust, len(self.X[0])))
        # print(self.mu.shape)
        for k in range(self.nb_clust):
            self.mu[k] = rd.choice(self.X)

        self.ci = np.zeros((len(self.X)))
        self.nk = np.zeros((nb_clust))

    def KMeans(self):
        LX = len(self.X)  # nb of datapoints

        ### do the following steps for self.nb_iter iterationsb ###

        for iter in range(self.nb_iter):
            centerslist = []
            print('iteration %s' % iter)

            # calculate for each cluster the measured Euclidian distance between x_i and mu_k
            # and select for c_i the cluster with the minimal distance
            count = 0
            for xi in self.X:
                val1 = 1e10
                clust = -1
                for k in range(self.nb_clust):
                    mu = self.mu[k]
                    val2 = np.dot(xi - mu, xi - mu)
                    # if count == 133:
                    # print('xi: %s, mu: %s, xi-mu: %s' % (xi, mu, xi - mu))

                    if val2 < val1:
                        val1 = val2
                        clust = k
                        # if count == 133:
                        # print('count: %s, k:%s, np.dot:%s' % (count, k, val2))
                self.ci[count] = clust
                count += 1

            # calculate numbers of indicators c_i for each cluster k
            for k in range(self.nb_clust):
                nbk = 0
                for i in range(LX):
                    if self.ci[i] == k:
                        nbk += 1
                self.nk[k] = nbk
                if self.nk[k] == 0:
                    self.nk[k] = 1
            # print('nk: %s'%self.nk)

            # Update the mean mu = (mu_1,...,m_K)
            for k in range(self.nb_clust):
                val = np.zeros((len(self.X[0])))
                count = 0
                for i in range(LX):
                    if self.ci[i] == k:
                        val += self.X[i]
                self.mu[k] = val / self.nk[k]
                centerslist.append(self.mu[k])

            # save centroids into file for each iteration
            filename = "centroids-" + str(iter + 1) + ".csv"  # "i" would be each iteration
            np.savetxt(filename, centerslist, delimiter=",")
            # print(Kmeans.check_prediction(self, self.ci))

        return self.mu, self.ci

    def check_prediction(self, ci):
        LX = len(self.X)  # nb of datapoints
        cTT = 50
        ct1 = 0
        ct2 = 0
        ct3 = 0
        print(ci)
        for i in range(LX):
            if 0 <= i < cTT:
                if i == 0:
                    val = ci[i]
                if ci[i] == val:
                    ct1 += 1
            elif cTT <= i < 2 * cTT:
                if i == cTT:
                    val = ci[i]
                if ci[i] == val:
                    ct2 += 1
            elif 2 * cTT <= i < 3 * cTT:
                if i == 2 * cTT:
                    val = ci[i]
                if ci[i] == val:
                    ct3 += 1

        return ct1, ct2, ct3

    def plot_data(self, ci, mu, nb):
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        x3 = []
        y3 = []
        x4 = []
        y4 = []
        x5 = []
        y5 = []
        mux = []
        muy = []

        # get mean
        for k in range(len(mu)):
            mux.append(mu[k][0])
            muy.append(mu[k][1])

        ## plot datapoints per cluster
        colors = ('b', 'r', 'g', 'c', 'y')
        groups = ("cluster 1", "cluster 2", "cluster 3", "cluster 4", "cluster 5")
        for i in range(len(self.X)):
            if ci[i] == 0:
                x1.append(self.X[i][0])
                y1.append(self.X[i][1])
            elif ci[i] == 1:
                x2.append(self.X[i][0])
                y2.append(self.X[i][1])
            elif ci[i] == 2:
                x3.append(self.X[i][0])
                y3.append(self.X[i][1])
            elif ci[i] == 3:
                x4.append(self.X[i][0])
                y4.append(self.X[i][1])
            elif ci[i] == 4:
                x5.append(self.X[i][0])
                y5.append(self.X[i][1])

        g1 = (x1, y1)
        g2 = (x2, y2)
        g3 = (x3, y3)
        g4 = (x4, y4)
        g5 = (x5, y5)
        data = (g1, g2, g3, g4, g5)

        # Create plot
        fig = plt.figure()
        #ax = fig.add_subplot(1, 1, 1, axisbg="1.0") #python 2.7
        ax = fig.add_subplot(1, 1, 1)
        for data, color, group in zip(data, colors[0:nb], groups[0:nb]):
            x, y = data
            ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
        plt.scatter(mux, muy, c='k', marker='x', s=100)
        plt.legend(loc=2)
        plt.show()


class GMM:
    def __init__(self, nb_clust, nb_iter):
        self.X = np.genfromtxt(sys.argv[1], delimiter=",")
        self.nb_clust = nb_clust
        self.nb_iter = nb_iter
        self.nk = np.zeros((nb_clust))

        # Random initialisation of the mean mu = (mu_1,...,m_K)
        self.mu = np.zeros((nb_clust, len(self.X[0])))
        for k in range(self.nb_clust):
            self.mu[k] = rd.choice(self.X)

        # Initialisation of the sigma_k matrix as identity matrix
        self.sigma = np.zeros((nb_clust, len(self.X[0]), len(self.X[0])))
        for k in range(self.nb_clust):
            self.sigma[k] = np.matrix(np.identity(len(self.X[0])), copy=False)

        # Initialisation of the conditional posterior probability distribution
        self.phi = np.zeros((len(self.X), nb_clust))

        # Initialisation of pi as a uniform distribution
        self.pi = np.random.uniform(low=0.0, high=1.0, size=nb_clust)

    def EMGMM(self):
        LX = len(self.X)  # nb of datapoints
        L_save = []  # list saving loss function at each iteration

        ### do the following steps for self.nb_iter iterationsb ###

        for iter in range(self.nb_iter):
            print("iteration: %s" % (iter + 1))
            # Initialize objective function at the beginning of each iteration
            L = 0

            ## E-step: generate phi posterior probability
            # print("E-step")
            # calculate for each datapoint x_i the sum of pi_k*N_k(mu_k,Sigma_k) to use later for normalization
            sum_pi_j_N = np.zeros((len(self.X)))
            for i in range(LX):
                val = 0
                for k in range(self.nb_clust):
                    val += self.pi[k] * MvN.pdf(self.X[i], mean=self.mu[k], cov=self.sigma[k])
                    # print(MvN.pdf(self.X[i], mean=self.mu[k], cov=self.sigma[k]))
                sum_pi_j_N[i] = val

            # calculate phi & objective function
            L = 0  # Initialize objective function at the beginning of each iteration
            for i in range(LX):
                val2 = 0
                for k in range(self.nb_clust):
                    val = self.pi[k] * MvN.pdf(self.X[i], mean=self.mu[k], cov=self.sigma[k])
                    val2 += val
                    self.phi[i][k] = val / sum_pi_j_N[i]
                L += np.log(val2)
            L_save.append(L)

            ## M-step
            # print("M-step")
            # update empirical distribution pi_k using expected nb of pts n_k coming from cluster k
            for k in range(self.nb_clust):
                val = 0
                for i in range(LX):
                    val += self.phi[i][k]
                self.nk[k] = val
                if self.nk[k] == 0:
                    self.nk[k] = 1

            for k in range(self.nb_clust):
                self.pi[k] = self.nk[k] / LX

            # update mean mu_k
            for k in range(self.nb_clust):
                val = np.zeros((len(self.X[0])))
                for i in range(LX):
                    # print(self.phi[i][k],self.X[i])
                    val += self.phi[i][k] * self.X[i]
                self.mu[k] = val / self.nk[k]

            # update covariance sigma_k
            for k in range(self.nb_clust):
                val = np.zeros((len(self.X[0]), len(self.X[0])))
                for i in range(LX):
                    diff_iT = self.X[i] - self.mu[k]
                    diff_i = diff_iT[np.newaxis, :].T
                    aa = np.zeros((len(self.X[0]), len(self.X[0])))
                    bb = np.zeros((len(self.X[0]), len(self.X[0])))
                    bb[0] = diff_iT
                    for j in range(len(self.X[0])):
                        aa[j][0] = diff_i[j]

                    val += self.phi[i][k] * np.dot(aa, bb)
                self.sigma[k] = val / self.nk[k]

            # save pi, phi, mu & sigma into files for each iteration
            filename = "pi-" + str(iter + 1) + ".csv"
            np.savetxt(filename, self.pi, delimiter=",")
            filename = "phi-" + str(iter + 1) + ".csv"
            np.savetxt(filename, self.phi, delimiter=",")
            filename = "mu-" + str(iter + 1) + ".csv"
            np.savetxt(filename, self.mu, delimiter=",")  # this must be done at every iteration

            for k in range(self.nb_clust):  # k is the number of clusters
                filename = "Sigma-" + str(k + 1) + "-" + str(
                    iter + 1) + ".csv"  # this must be done 5 times (or the number of clusters) for each iteration
                np.savetxt(filename, self.sigma[k], delimiter=",")

        # save objective function
        np.savetxt("Loss_function.csv", L_save, delimiter=",")

        # Plot objective function
        plt.plot(L_save, linewidth=3)
        axes = plt.gca()
        # axes.set_ylim([-0.5e6, 0])
        plt.ticklabel_format(axis='y', style='sci')  # , scilimits=(-2, 2))
        plt.grid()
        plt.xlabel('Iterations', fontsize=20)
        plt.ylabel('Loss function', fontsize=20)
        axes.tick_params(labelsize=14)
        plt.show()

        return self.pi, self.mu, self.sigma, self.phi

    def plot_GMM(self, phi, nb):

        # create two vectors with data point coordinates (we are considering to be in 2D)
        x = []
        y = []
        for i in range(len(self.X)):
            x.append(self.X[i][0])
            y.append(self.X[i][3])

        # adjust phi for colors
        col = phi[:]
        for i in range(len(self.X)):
            for k in range(nb):
                if col[i][k] < 0.01:
                    col[i][k] = 0
                elif col[i][k] > 1:
                    col[i][k] = 1
                    # else:
                    #     col[i][k] *= 1
                    # col[i][k] = int(round(col[i][k]))

        np.savetxt("colors.csv", col, delimiter=",")

        groups_colors = {'cluster 1': 'r', 'cluster 2': [0, 1, 0], 'cluster 3': 'b'}
        # Create plot
        fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, axisbg="1.0") # python 2.7
        ax = fig.add_subplot(1, 1, 1)
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(colors, groups)
        patchList = []
        for key in groups_colors:
            data_key = mpatches.Patch(color=groups_colors[key], label=key)
            patchList.append(data_key)

        ax.legend(handles=patchList, loc=2)

        for i in range(len(self.X)):
            ax.scatter(x[i], y[i], alpha=0.8, c=[col[i][0], col[i][1], col[i][2], ], edgecolors='none',
                       s=40)  # , label=group)

        plt.xlabel('Attribute 1', fontsize=18)
        plt.ylabel('Attribute 4', fontsize=18)
        ax.tick_params(labelsize=14)

        plt.show()


    def plot_GMM_3D(self, phi, nb):
        # create two vectors with data point coordinates (we are considering to be in 3D)
        x = []
        y = []
        z = []

        for i in range(len(self.X)):
            x.append(self.X[i][0])
            y.append(self.X[i][1])
            z.append(self.X[i][2])

        # adjust phi for colors
        col = phi[:]
        for i in range(len(self.X)):
            for k in range(nb):
                if col[i][k] < 0.01:
                    col[i][k] = 0
                elif col[i][k] > 1:
                    col[i][k] = 1


        np.savetxt("colors.csv", col, delimiter=",")

        groups_colors = {'cluster 1': 'r', 'cluster 2': [0, 1, 0], 'cluster 3': 'b'}
        # Create plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        patchList = []
        for key in groups_colors:
            data_key = mpatches.Patch(color=groups_colors[key], label=key)
            patchList.append(data_key)

        ax.legend(handles=patchList, loc=2)

        for i in range(len(self.X)):
            ax.scatter(x[i], y[i], z[i], zdir='z', s=50, c=[col[i][0], col[i][1], col[i][2], ], edgecolors='none')

        ax.set_xlabel('Attribute 1', fontsize=18)
        ax.set_ylabel('Attribute 2', fontsize=18)
        ax.set_zlabel('Attribute 3', fontsize=18)
        tick_spacing = 1.0
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.tick_params(labelsize=12)

        plt.show()


if __name__ == "__main__":
    try:
        assert len(sys.argv) > 5, "Missing input arguments"
        iter = int(sys.argv[5])  # nb of iterations
        nb = int(sys.argv[3])    # nb of clusters
        Cl_type = sys.argv[4]    # type of clustering
        if Cl_type == "Kmeans":
            mu, ci = Kmeans(nb, iter).KMeans()
            ct1, ct2, ct3 = Kmeans(nb, iter).check_prediction(ci)
            Kmeans(nb, iter).plot_data(ci, mu, nb)
            print("ct1 = %s; ct2 = %s; ct3 = %s" % (ct1, ct2, ct3))
        elif Cl_type == "GMM":
            pi, mu, sigma, phi = GMM(nb, iter).EMGMM()
            GMM(nb, iter).plot_GMM(phi, nb)
            GMM(nb, iter).plot_GMM_3D(phi, nb)
        else:
            print("Clustering type not known")
    except Exception as e:
        print(e)
