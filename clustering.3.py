# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:20:25 2015

@author: school
"""

import os
import numpy
#from sklearn.cluster import k_means
from scipy.cluster.vq import whiten
import matplotlib
import matplotlib.pyplot as plt
#from statsmodels.sandbox.regression.predstd import wls_prediction_std
import regression
from Bio.Cluster import distancematrix, kmedoids
from collections import Counter
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import ward, dendrogram


clear = lambda: os.system('clear')
clear()

groups = 8
#samples = [2,3,4,5,6,7,8,9,10,11,13]
samples = open("FDC.csv", "r"). read().split(",")
samples = [ float(value) for value in samples]


"""
diff_samples = numpy.array(original_samples + [0])-numpy.array([0] + original_samples)
diff_samples = list(diff_samples)
difff_samples = numpy.array((diff_samples + [0])) - numpy.array([0] + diff_samples)
"""


#執行階層式分群作業
tsamples = numpy.array([samples]).transpose()
distance = distance_matrix(tsamples, tsamples)
hc = ward(distance)

#print(hc)
dendrogram(hc)

def find_majority( array, index):
    if index < 5:
        start = 0
    else:
        start = index - 5
    
    end = index + 6
    datas = array[start:end]
    
    counter = Counter(datas)
    [(majority, count)] = counter.most_common(1)
    
    return majority
    
def majority_filter(array):
    filter_array = [find_majority(array, index) for index in range(len(array))]
    return filter_array
    
def plots_outlier(samples, outlier_x):
    outlier_y = [samples[x-1] for x in outlier_x]    
    
    ax.plot(outlier_x, outlier_y, 'or')


Y = samples
samples = [(x, sample) for x, sample in enumerate(samples, start=1)]

distance =  distancematrix(samples)



clusterid, error, nfound = kmedoids(distance, nclusters=groups, npass=10)

clusterid = majority_filter(clusterid)

"""
#對資料做kmeans    
input = numpy.array(samples)
whitened = whiten(input)    
___, labels, ___ = k_means(X = whitened, n_clusters = groups)
"""

segments = [list() for i in range(len(clusterid)+1)] #產生五個新的獨立list


#將不同群的資料放到不同列
for clusteri, sample in zip(clusterid, samples):
    segments[clusteri].append(sample)


clean_segments = [points for points in segments if len(points) >= 10]

    
    
num_spc = len(clean_segments)

def elimate_outlier(x, y):
    slope, intercept, std = regression.linear(y=y, x=x)
    center = regline(x=x, slope=slope, intercept=intercept)
  
    
    is_valid = abs(y-center) < 3*std 
    
    new_x = [value for value, is_valid in zip(x, is_valid) if is_valid]
    new_y = [value for value, is_valid in zip(y, is_valid) if is_valid]
    outlier = [value for value, is_valid in zip(x, is_valid) if not is_valid]
    
    if len(outlier) > 0:
        new_x, new_y, outlier2 = elimate_outlier(new_x, new_y)
        outlier = outlier2 + outlier
    
    return new_x, new_y, outlier
    

def regline(x, slope, intercept):
    x =numpy.array(x)
    y = intercept + slope*x
    return y

def regspc(x,y):
    slope, intercept, std = regression.linear(y=y, x=x)
    center = regline(x=x, slope=slope, intercept=intercept)
    upper = center + 3*std
    lower = center - 3*std
    
    return upper, lower, center
    
    

fig, ax = plt.subplots(figsize=(8,6))


uppers = []
lowers = []
centers = []
xs = []


for seg in clean_segments:
    y=[y for x,y in seg ]
    x=[x for x,y in seg ]
    x, y, outlier = elimate_outlier(x, y)
    plots_outlier(Y, outlier)
    
    
    upper, lower, center = regspc(x,y)
    uppers.extend(upper)
    lowers.extend(lower)
    centers.extend(center)
    xs.extend(x)
    

    ax.plot(x, y, 'ob')


ax.plot(xs, centers, 'b-')
ax.plot(xs, uppers, 'k-')
ax.plot(xs, lowers, 'k-')

legend = ax.legend(loc="best")

    


