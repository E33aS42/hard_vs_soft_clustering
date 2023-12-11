# hard_vs_soft_clustering
Project comparing 2 different clustering methods, hard(Kmeans) vs. soft (Gaussian Mixture Model).

# Introduction

Clustering analysis is a unsupervised learning technique that is applied on an unlabelled database to identify sets of objects that belong to groups of similar properties or interests. It is an application used in many fields including image analysis, bioinformatics, social networks, social science, and so on...

The objective in this project is apply two types of clustering analysis:
• a hard-clustering analysis, the K-means model.
• a type of soft-clustering analysis, the Gaussian mixture model.

The difference between hard and soft clustering is that, in one case, every single data point is assigned to
only one cluster and boundaries are clearly visible, while in the other case, a weight is applied to each data
point telling how confident we are that it belongs to a given cluster. As such, boundaries appear more fuzzy.

The most popular database for education purposes, the Iris dataset [1], will be used for the analysis. It
contains 150 datapoints each having 4 different attributes and belonging to one of three different classes.
The goal will be to identify the class of each datapoint using a clustering model.

[1] https://archive.ics.uci.edu/ml/index.php


A detailed presentation of the project realization can be read in the clustering_report.pdf file.
