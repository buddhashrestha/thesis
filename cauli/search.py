import numpy
import pandas as pd
import faiss
from cauli.utils.segments import Segments
from cauli.findVideos import findVideos
import ast
import json
from bitmap.src.bitmap import BitMap
from collections import deque
import datetime
#this comes from p1 and p2


def search(q,query_type):
    videos = findVideos(q)
    timings = {}
    for each_video in videos:
        df_embeddings = pd.read_csv("data/"+str(each_video) + '/person_embeddings_mapping.csv',sep='\t')
        cols = list(df_embeddings)
        cols.insert(0, cols.pop(cols.index('Embeddings')))
        df_embeddings = df_embeddings.ix[:, cols]

        x = str(df_embeddings['Embeddings'].tolist()).replace("\'", "")
        x = ast.literal_eval(x)
        y = numpy.array(x)
        y = y.astype('float32')
        d = 128
        nlist = 1
        k = 1
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)



        index.train(y)  # t and y ma k farak cha?

        index.add(y)  # add may be a bit slower as well

        D, I = index.search(q, k)  # actual search
        pos = [0] * len(I)
        p = [0] * len(I)
        # if face is not present: then add to the list
        if I == [[]]:
            print("Not found")

        else:
            for i in range(len(I)):
                pos[i] = I[i][0]
                p[i] = df_embeddings.iloc[pos[i], 1]

        df_person_bitmap = pd.read_csv("data/"+str(each_video) +'/person_bitmap_vector.csv',sep='\t')
        df_person_bitmap = df_person_bitmap.loc[:, ~df_person_bitmap.columns.str.contains('^Unnamed')]
        person_bitmap = [0] * len(p)



        for i in range(len(p)):
            x = df_person_bitmap.loc[df_person_bitmap['person_label'] == int(p[i]), df_person_bitmap.columns != 'person_label'].values[0]
            x[numpy.isnan(x)] = 0
            person_bitmap[i] = x.astype(int)
            person_bitmap[i] = "".join(map(str, person_bitmap[i]))
            person_bitmap[i] = int(person_bitmap[i], 2)
            # person_bitmap[i] = numpy.array(person_bitmap,dtype=numpy.uint32)
        a = 0b0000111
        b= 0b0000111
        import timeit
        # print("Time taken to do and :", timeit.timeit('x = 0b11111100000001111001; d = 0b11111100000001111001; e =0b11111100000001111001;f =0b11111100000001111001;g =0b11111100000001111001; c = x & d & e & f&g', number=10000))
        # exit(0)
        df_segment = pd.read_csv("data/" + str(each_video) + '/person_segment.csv', sep='\t')

        l = []
        for each_i in df_segment['segment']:
            start = each_i.split(":")[0]
            end = each_i.split(":")[1]
            l.append([start,end])
        # exit(0)
        a = datetime.datetime.now()

        if query_type == 'next':
            timings[each_video] = next(person_bitmap[0],person_bitmap[1])
        if query_type == 'eventually':
            timings[each_video] = eventually(person_bitmap[0], person_bitmap[1])
        if query_type == 'is_before':
            timings[each_video] = is_a_before_b(person_bitmap[0],person_bitmap[1])
        if query_type == 'interval':
            timings[each_video] = interval(person_bitmap,l)

        b = datetime.datetime.now()
        c = b - a
        print("Time required to extract segments -----------> : ", c.microseconds)

    # timings = convert_segment_to_timings(timings)
    return timings


def interval(person_bitmap,l):#total 30us
#################################
    # c = numpy.prod(person_bitmap)

    c = numpy.array(person_bitmap,dtype=numpy.uint32)
    c = c.all(axis=0)
    x = numpy.where(c)[False]

################################# 13us
    #calculate segment now
    c = Segments()
    segs = c.find_continous_segments(x,l)

##################################
    return segs

def convert_segment_to_timings(video_segments,df_segment):
    timings = {}
    each_segment = []
    for start,end in video_segments:
        start = df_segment.loc[df_segment['index'] == start]['segment'].values[0]
        start = start.split(":")[0]
        end = df_segment.loc[df_segment['index'] == end]['segment'].values[0]
        end = end.split(":")[1]
        each_segment.append([start,end])
    return each_segment

def convert_segment_to_timings1(video_segments):
    timings = {}
    for each_video in video_segments:
        df_segment = pd.read_csv("data/" + str(each_video) + '/person_segment.csv', sep='\t')
        video_seg = video_segments[each_video]
        each_segment = []
        for start,end in video_seg:
            start = df_segment.loc[df_segment['index'] == start]['segment'].values[0]
            start = start.split(":")[0]
            end = df_segment.loc[df_segment['index'] == end]['segment'].values[0]
            end = end.split(":")[1]
            each_segment.append([start,end])
        timings[each_video] = each_segment
    return timings

def next(p1,p2):
    p1_list = p1
    p2_list = deque(p2)

    p2_list.popleft()
    p2_list = list(p2_list)

    p1_list = p1_list[:len(p1_list)-2]

    final = [0] * len(p1_list)
    for i in range(len(p1_list)-1):
        final[i] = p1_list[i] *  p2_list[i]

    vector = "".join(str(v) for v in final)
    bm = BitMap.fromstring(vector)
    int_p = int(bm.tostring(), 2)
    if (int_p > 0):
        return True
    else:
        return False

def eventually(p1, p2):
    # bm_p1  a = datetime.datetime.now()= BitMap.fromstring(p1)
    # bm_p2 = BitMap.fromstring(p2)
    p1_list = deque(p1)
    p2_list = deque(p2)
    for a in p1:
        if a != 1:
            p1_list.popleft()
            p2_list.popleft()
        else:
            break
    p1_list = list(p1_list)
    p2_list = list(p2_list)

    p1_vector = "".join(str(v) for v in p1_list)
    p2_vector = "".join(str(v) for v in p2_list)
    p1_bm = BitMap.fromstring(p1_vector)
    p2_bm = BitMap.fromstring(p2_vector)
    int_p1 = int(p1_bm.tostring(),2)
    int_p2 = int(p2_bm.tostring(),2)
    if (int_p1>int_p2):
        return True
    else:
        return False

def is_a_before_b(p1,p2):
    p1_vector = "".join(str(v) for v in p1)
    p2_vector = "".join(str(v) for v in p2)
    p1_bm = BitMap.fromstring(p1_vector)
    p2_bm = BitMap.fromstring(p2_vector)
    int_p1 = int(p1_bm.tostring(), 2)
    int_p2 = int(p2_bm.tostring(), 2)
    if (int_p1 > int_p2):
        return True
    else:
        return False


q = []
file_path = "/home/buddha/thesis/cauli/test/person_embeddings.csv"
df_matrix = pd.read_csv(file_path, sep='\t')

#no coappearance:
my_list = [['leonard.jpg', 'rachel.jpg'],
           ['leonard.jpg', 'ross.jpg'],
           ['leonard.jpg', 'monica.jpg'],
           ['leonard.jpg', 'phoebe.jpg'],
           ['leonard.jpg', 'chandler.jpg'],
           ['leonard.jpg', 'joey.jpg'],
           ['sheldon.jpg', 'rachel.jpg'],
           ['sheldon.jpg', 'ross.jpg'],
           ['sheldon.jpg', 'monica.jpg'],
           ['sheldon.jpg', 'phoebe.jpg'],
           ['sheldon.jpg', 'chandler.jpg'],
           ['sheldon.jpg', 'joey.jpg'],
           ['howard.jpg', 'rachel.jpg'],
           ['howard.jpg', 'ross.jpg'],
           ['howard.jpg', 'monica.jpg'],
           ['howard.jpg', 'phoebe.jpg'],
           ['howard.jpg', 'chandler.jpg'],
           ['howard.jpg', 'joey.jpg'],
           ['raj.jpg', 'rachel.jpg'],
           ['raj.jpg', 'ross.jpg'],
           ['raj.jpg', 'monica.jpg'],
           ['raj.jpg', 'phoebe.jpg'],
           ['raj.jpg', 'chandler.jpg'],
           ['raj.jpg', 'joey.jpg']
           ]
#co-appearance but query not satisfied
my_list = [['sheldon.jpg', 'leonard.jpg'],
           ['sheldon.jpg', 'raj.jpg'],
           ['leonard.jpg', 'raj.jpg'],
           ['chandler.jpg','monica.jpg'],
           ['monica.jpg','phoebe.jpg'],
           ['chandler.jpg','monica.jpg'],
           ['joey.jpg','monica.jpg']
           ]
# #coappearance and query satisfied
# my_list = [['sheldon.jpg', 'howard.jpg'],
#            ['penny.jpg', 'howard.jpg'],
#            ['penny.jpg', 'raj.jpg'],
#            ['penny.jpg', 'leonard.jpg'],
#            ['chandler.jpg','ross.jpg'],
#            ['chandler.jpg','rachel.jpg'],
#            ['joey.jpg','phoebe.jpg'],
#            ]
my_list = [['ross.jpg','rachel.jpg']]
my_list = [['bernadette.jpg','raj.jpg','penny.jpg','howard.jpg','leonard.jpg']]
my_list = [['raj.jpg','penny.jpg','howard.jpg']]


for each_pair in my_list:
    q = []
    p = {}
    print(each_pair[0])

    for i,each_person in enumerate(each_pair):
        print(i)
        p[i] = each_pair[i]
        print("Person : ", p[i])
        p[i] = df_matrix.loc[df_matrix['person'] == p[i]]['emb'].values[0]
        p[i] = ast.literal_eval(p[i])
        p[i] = numpy.array(p[i], dtype=float)
        q.append(p[i])

    q = numpy.array(q)
    q = q.astype('float32')

    print("Search : ",search(q,"interval"))
'''
#TO debug
enter your query person list
goto findVideos.py

add these lines after person search:
print(df.iloc[pos[i],0])
exit(0) 

then take an element of the array
compare that with person_embeddings.csv and check if it is there or not
also check in person_to_video_matrix.csv and find the row number in it. then check the videos that person appears in
it will give you alot of idea of what could have went wrong!!!


'''
#TODO:loadup new video, run a person search on a custom photo using FaceDescriptor