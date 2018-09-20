import faiss
from pandas import read_table
from pyannoteLib.core import Segment, Annotation
import numpy as np
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import DBSCAN
import dlib
import glob
import os
import cv2
import pandas
def _to_segment(group):
    return Segment(np.min(group.time), np.max(group.time))



# data_path = "../data/"+ str(5) + "/"+  "bbt1_720.embedding.txt"
data_path = "../data/"+ str(1) + "/"+  "bbt1.embedding.txt"
#
names = ['time', 'track']

for i in range(128):
    names += ['d{0}'.format(i)]
#
data = read_table(data_path, delim_whitespace=True,
                  header=None, names=names)

data.sort_values(by=['track', 'time'], inplace=True)
#
descriptors = []
x = data.iloc[:, 2:].values
for each_i in x:
    fd = dlib.vector(each_i)
    descriptors.append(fd)
#
thresh = 1.5


labels = dlib.chinese_whispers_clustering(descriptors, 0.5)

# determine the total number of unique faces found in the dataset
labelIDs = np.unique(labels)
print(labelIDs)

data['cluster'] = pandas.Series(labels, index=data.index)
print(len(data.loc[data['cluster']==1]))
track_cluster = data.groupby(by='track', as_index=False).first()[['track','cluster']].values
# print(track_cluster)
print(track_cluster[:][:,[1]])
starting_point = Annotation(modality='face')
for track, segment in data.groupby('track').apply(_to_segment).iteritems():
    if not segment:
        continue
    starting_point[segment, track] = track_cluster[track][1]

print(starting_point.label_timeline(2))
for each_i in starting_point.itersegments():
    # print(starting_point.get_labels(each_i))
    # print(starting_point.get_tracks(each_i))
    start,end = each_i
    # print("Start:",start, "End:",end)
