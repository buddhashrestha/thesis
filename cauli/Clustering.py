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



class Clustering(object):
    def __init__(self,data_path):
        names = ['time', 'track']

        for i in range(128):
            names += ['d{0}'.format(i)]
        #
        self.data = read_table(data_path, delim_whitespace=True,
                          header=None, names=names)

        self.data.sort_values(by=['track', 'time'], inplace=True)

        descriptors = []
        x = self.data.iloc[:, 2:].values
        for each_i in x:
            fd = dlib.vector(each_i)
            descriptors.append(fd)
        #
        thresh = 1.5

        labels = dlib.chinese_whispers_clustering(descriptors, 0.5)

        # self.labels = np.unique(labels)


        self.data['cluster'] = pandas.Series(labels, index=self.data.index)
        # print(len(self.data.loc[self.data['cluster'] == 1]))
        #TODO: this can be improved by taking highest count of label in each track
        track_cluster = self.data.groupby(by='track', as_index=False).first()[['track', 'cluster']].values

        self.labels = np.unique(track_cluster[:][:,[1]])
        self.starting_point = Annotation(modality='face')
        for track, segment in self.data.groupby('track').apply(_to_segment).iteritems():
            if not segment:
                continue
            self.starting_point[segment, track] = track_cluster[track][1]



    def get_labels(self):
        return self.labels

    def get_embeddings(self):
        return self.data


    def get_starting_point(self):
        return self.starting_point

def _to_segment(group):
    return Segment(np.min(group.time), np.max(group.time))



