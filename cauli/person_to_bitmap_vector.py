import os,sys
sys.path.append('../')
from pyannoteVideo.pyannote.video.face.clustering import FaceClustering
import pandas as pd
import numpy as np

from utils import file

clustering = FaceClustering(threshold=0.6)
face_tracks, embeddings = clustering.model.preprocess('../pyannote-data/TheBigBangTheory.embedding.txt')

result = clustering(face_tracks, features=embeddings)

current_directory = os.getcwd()


#video duration in seconds
video_duration = int(embeddings['time'].iloc[-1])

ar = [0] * video_duration
embeddings_labels_mappings = {}
person = {}
for each_label in result.labels():
    embeddings_labels_mappings[each_label] = list(embeddings.loc[embeddings['track'] == each_label].iloc[0, 2:])
    for each_segment in result.itersegments():
        if each_label == result.get_labels(each_segment).pop():
            for i in range(int(each_segment._get_start()),int(each_segment._get_end())):
                ar[i] = 1
    person[each_label] = ar
    ar = [0] * video_duration
print(person)

df_person = pd.DataFrame(list(person.items()), columns=['person_label', 'BitMap'])
df_embeddings = pd.DataFrame(list(embeddings_labels_mappings.items()), columns=['person_label', 'Embeddings'])

vid_num = 1

file.check_directory(current_directory + "/" + str(vid_num))

df_person.to_csv(current_directory + "/" + str(vid_num) + '/person_bitmap_vector.csv', sep='\t')
df_embeddings.to_csv(current_directory + "/" + str(vid_num) + '/person_embeddings_mapping.csv', sep='\t')


import numpy
import pandas as pd
import faiss
import ast
import os


d = 128
data_matrix = current_directory + '/data/person_to_video_matrix.csv'
if not(os.path.exists(data_matrix)):
    df = pd.DataFrame(columns=['person','0'])
    df[0] = 0
    video_num = 1
    for each_label in result.labels():
        q = list(embeddings.loc[embeddings['track'] == each_label].iloc[0, 2:])
        q = q.astype('float32')
        q = q.reshape(1, 128)
        # if face is not present: then add to the list
        df2 = pd.DataFrame({'person': q.tolist(), video_num: 1})
        df = pd.concat([df, df2])
else:
    df = pd.read_csv(data_matrix, sep='\t')

    cols = list(df)
    cols.insert(0, cols.pop(cols.index('person')))
    df = df.ix[:, cols]

    x = str(df['person'].tolist()).replace("\'","")
    x = ast.literal_eval(x)
    y = numpy.array(x)
    y = y.astype('float32')
    #now you can train x as you did t

    #add new column for that video
    #cols = pd.read_csv('test.csv',sep='\t').columns.tolist()
    video_num = int(cols[-1]) + 1

    #by default every video is zero
    df[video_num] = 0

    #searching section

    nlist = 1
    k = 1
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)


    index.train(y)# t and y ma k farak cha?

    index.add(y)                  # add may be a bit slower as well

    for each_label in result.labels():
        q = list(embeddings.loc[embeddings['track'] == each_label].iloc[0, 2:])
        q = q.astype('float32')
        q = q.reshape(1, 128)
        D, I = index.search(q, k)     # actual search
        #if face is not present: then add to the list
        if I ==[[]]:
            df2 = pd.DataFrame({'p':q.tolist(),video_num:1})
            df = pd.concat([df,df2])
        else:
            pos = I[0][0]
            df.iloc[pos, df.columns.get_loc(video_num)] = 1

df.to_csv(current_directory + '/data/person_to_video_matrix.csv', sep='\t')


