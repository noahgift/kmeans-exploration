# -*- coding: utf-8 -*-
# K-means Cluster Analysis (Python)

# programmed by Thomas W. Miller, August 2017

from __future__ import division, print_function

# import packages for this example
import pandas as pd  # DataFrame operations 
from collections import OrderedDict  # to create DataFrame with ordered columns
# special plotting methods
from pandas.tools.plotting import scatter_matrix    
import numpy as np  # arrays and math functions
import matplotlib.pyplot as plt  # static plotting
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics  # for silhouette coefficient

# previously obtained data from public-domain source at Stanford University
# 240 student participants' self-ratings on 32 personality attributes
# a review of these data suggests that student survey participants were
# given an adjective check-list with instructions to self-rate such as:
# "Rate the extent to which each adjective describes you. Use a 
# 1-to-9 scale, where 1 means 'very much unlike me' and 
# 9 means 'very much like me.' " 
# source: http://www.stanford.edu/class/psych253/data/personality0.txt
# in previous work with R, we created a csv file from these data 

# create Pandas DataFram from the student data
# define a pandas DataFrame
student_data = pd.read_csv("student_data.csv")

# for fun, consider adding your own data to the student_data
# data frame by self-rating the 32 adjectives on a 1-to-9 scale ...
# this would provide 241 observations
# for example...
# my_data_dict = {"distant":1, "talkative":5, "careless":1, "hardworking":8, 
#   "anxious":2, "agreeable":6, "tense":1, "kind":7, "opposing":3, "relaxed":5,
#   "disorganized":4, "outgoing":5, "approving":3, "shy":1, "disciplined":5, 
#   "harsh":1, "persevering":9, "friendly":7, "worrying":3, "responsive":6,
#   "contrary":2, "sociable":6, "lazy":1, "cooperative":8, "quiet":3,   
#   "organized":6, "critical":5, "lax":2, "laidback":5, "withdrawn":1,
#   "givingup":1, "easygoing":6}
# my_data_frame = pd.DataFrame(my_data_dict, index = [0])
# student_data = pd.concat([student_data, my_data_frame])

print('')
print('----- Summary of Input Data -----')
print('')

# show the object is a DataFrame
print('Object type: ', type(student_data))

# show number of observations in the DataFrame
print('Number of observations: ', len(student_data))

# show variable names
variable = student_data.columns
print('Variable names: ', variable)

# show descriptive statistics
pd.set_option('display.max_columns', None)  # do not limit output
print(student_data.describe())

# show a portion of the beginning of the DataFrame
print(student_data.head())

print('')
print('----- K-means Cluster Analysis of Variables -----')
print('')

# it is good practice to standardize variables prior to clustering
# work with standard scores for all cluster variables
# standard scores have zero mean and unit standard deviation
# here we standardize each student's data 
standardized_student_data_matrix = preprocessing.scale(student_data)

# transpose of matrix needed for clusters of variables
variable_cluster_data = standardized_student_data_matrix.T

# specify the number of clusters in order to perform 
# K-means cluster analysis on the variables in the study

# there is much psychological research about what are called
# the big five factors of perosnality:
# extraversion, agreeableness, conscientiousness, neuroticism, openness
#
# some personality researchers have focused on only two factors:
# extraversion/introversion and neuroticism

# suppose we think five factors (and five clusters) will be sufficient 
# here we use our knowledge of the big-five personality factors
# assuming that there may well be five clusters to identify
    
kmeans = KMeans(n_clusters = 5, n_init = 25, random_state = 1)
kmeans.fit(variable_cluster_data)
cluster = kmeans.predict(variable_cluster_data)  # cluster ids for variables

# create pandas DataFrame for summarizing the cluster analysis results
variable_kmeans_solution = pd.DataFrame(OrderedDict([('cluster', cluster),
    ('variable', variable )]))

# print(variable_kmeans_solution)

# print results of variable clustering one cluster at a time
for cluster_id in sorted(variable_kmeans_solution.cluster.unique()):
    print()
    print(variable_kmeans_solution.loc[variable_kmeans_solution['cluster'] == \
        cluster_id])

# The silhouette coefficient is a useful general-purpose index
# for evaluating the strength of a clustering solution. The original
# reference is
# Peter J. Rousseeuw (1987). “Silhouettes: a Graphical Aid to the 
#     Interpretation and Validation of Cluster Analysis”. 
#     Computational and Applied Mathematics 20: 53–65. 
#     doi:10.1016/0377-0427(87)90125-7.
# larger positive values of the silhouette coefficient are preferred
# these indicate dense, well separated clusters
   
# evaluate the clustering solution using the silhouette coefficient
print('Silhouette coefficient for the five-cluster k-means solution: ', 
    metrics.silhouette_score(variable_cluster_data, cluster, 
        metric = 'euclidean'))   
        
# a low silhouette coefficient suggests that we may want to try
# kmeans with alternative values for the number of clusters 
# or perhaps this problem is not particularly well suited for cluster analysis                                 
                                                                                       
print('')
print('----- Selected K-means Cluster Analysis for Student Segments -----')
print('')

# here we are working in much the way we would in a market research
# study looking for market segments... here segments/clusters of students

# it is good practice to standardize variables prior to clustering
# work with standard scores for all cluster variables across students
# standard scores have zero mean and unit standard deviation for all variables
# these were computed earlier in working on the variable clustering
student_cluster_data = standardized_student_data_matrix

# specify the number of clusters in order to perform 
# K-means cluster analysis on the variables in the study
# with no preconceived notions about the number of student segments/clusters
# we search across various cluster analysis solutions defined 
# each individual k-means solution is defined by the argument n_clusters

# consider selecting a solution based on the silhouette coefficient
for nclusters in range(2,21): # search between 2 and 20 clusters/segments
    kmeans = KMeans(n_clusters = nclusters, n_init = 25, random_state = 1)
    kmeans.fit(student_cluster_data)
    segment = kmeans.predict(student_cluster_data)  # cluster ids for variables
    print('nclusters: ', nclusters, ' silhouette coefficient: ', 
        metrics.silhouette_score(student_cluster_data, segment, 
            metric='euclidean'))

# results suggest that a two-cluster/segment solution is best

print('')
print('----- Solution for Two Student Segments -----')
print('')

kmeans = KMeans(n_clusters = 2, n_init = 25, random_state = 1)
kmeans.fit(student_cluster_data)
segment = kmeans.predict(student_cluster_data)  # cluster index

# create pandas DataFrame for summarizing the cluster analysis results
# using OrderedDict to preserve the order of column names
student_kmeans_solution = pd.DataFrame(OrderedDict(
    [('student', range(0,len(student_cluster_data))),
    ('segment', segment)]))

# to interpret the results of the segmentation 
# we can review the original ratings data for the two clusters/segments

# merge/join the segment information with the original student data
student_segmentation_data = student_kmeans_solution.join(student_data)

# try printing the means for attributes within each segment
for segment_id in sorted(student_segmentation_data.segment.unique()):
    print()
    print('Attribute means for segment: ', segment_id)
    this_student_segment_data = student_segmentation_data[ \
        student_segmentation_data.segment == segment_id]
    attributes = this_student_segment_data.ix[:,'distant':'easygoing'].mean()    
    print(attributes) 
    