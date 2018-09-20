from cauli.test.find_intervals import *
import pandas as pd
import numpy as np
from cauli.findVideos import findVideos
import faiss
import datetime

def search(q,query_type):
    videos = findVideos(q)
    timings = {}
    for each_video in videos:
        df_embeddings = pd.read_csv("/home/buddha/thesis/cauli/data/"+str(each_video) + '/person_embeddings_mapping.csv',sep='\t')
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


        df_person_bitmap = pd.read_csv("/home/buddha/thesis/cauli/data/"+str(each_video) +'/person_segment_naive.csv',sep='\t')
        person_bitmap = [0] * len(p)
        # print(df_person_bitmap.loc[:])
        table = pd.read_csv("/home/buddha/thesis/cauli/data/" + str(each_video) + '/person_segment_naive.csv', sep='\t')


        x = table.loc[table['person'] == int(p[0]), table.columns != 'person']['segment'].iloc[0]
        x = ast.literal_eval(x)
        x = numpy.array(x, dtype=float)

        y = table.loc[table['person'] == int(p[1]), table.columns != 'person']['segment'].iloc[0]
        y = ast.literal_eval(y)
        y = numpy.array(y, dtype=float)

        z = table.loc[table['person'] == int(p[2]), table.columns != 'person']['segment'].iloc[0]
        z = ast.literal_eval(z)
        z = numpy.array(z, dtype=float)

        p = table.loc[table['person'] == int(p[3]), table.columns != 'person']['segment'].iloc[0]
        p = ast.literal_eval(p)
        p = numpy.array(p, dtype=float)

        # q1 = table.loc[table['person'] == int(p[4]), table.columns != 'person']['segment'].iloc[0]
        # q1 = ast.literal_eval(q1)
        # q1 = numpy.array(q1, dtype=float)
        #
        # r = table.loc[table['person'] == int(p[5]), table.columns != 'person']['segment'].iloc[0]
        # r = ast.literal_eval(r)
        # r = numpy.array(r, dtype=float)

        # a = datetime.datetime.now()
        # k = {}
        # k[0] = table.loc[table['person'] == int(p[0]), table.columns != 'person']['segment'].iloc[0]
        # k[0] = ast.literal_eval(k[0])
        # k[0] = numpy.array(k[0], dtype=float)
        # result = []
        # for i in range(1,len(p)-1):
        #     k[i] = table.loc[table['person'] == int(p[i]), table.columns != 'person']['segment'].iloc[0]
        #     k[i] = ast.literal_eval(k[i])
        #     k[i] = numpy.array(k[i], dtype=float)
        #     result = list(return_intersections(k[i], result))

        # print(x)
        # print(y)






        a = datetime.datetime.now()
        result = list(return_intersections(x, y))   #60us
        result = list(return_intersections(z,result))
        result = list(return_intersections(p,result))
        # result = list(return_intersections(q,result))
        # result = list(return_intersections(r,result))

        b = datetime.datetime.now()
        c = b - a

        print("Time required to find intersection : ", c.microseconds)
        timings[each_video] = result
    return timings


def interval(person_bitmap):
    table = pd.read_csv("/home/buddha/thesis/cauli/data/"+str(2) + '/person_segment_naive.csv',sep='\t')
    x = table.loc[table['person'] == int(18), table.columns != 'person']['segment'].iloc[0]
    y = table.loc[table['person'] == int(40), table.columns != 'person']['segment'].iloc[0]

    # x = np.fromstring(x[2:-2], sep=',').astype(float)
    #also can be done like this,
    x = ast.literal_eval(x)
    x = numpy.array(x, dtype=float)

    # y = np.fromstring(y[2:-2], sep=',').astype(float)
    #also can be done like this,
    y = ast.literal_eval(y)
    y = numpy.array(y, dtype=float)
    print("Y: ",y)
    print("X: ",x)


q = []
file_path = "/home/buddha/thesis/cauli/test/person_embeddings.csv"
df_matrix = pd.read_csv(file_path, sep='\t')


my_list = [['sheldon.jpg', 'howard.jpg'],
           ['penny.jpg', 'howard.jpg'],
           ['penny.jpg', 'raj.jpg'],
           ['penny.jpg', 'leonard.jpg'],
           ['chandler.jpg','ross.jpg'],
           ['chandler.jpg','rachel.jpg'],
           ['joey.jpg','phoebe.jpg'],
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
my_list = [['bernadette.jpg','raj.jpg','penny.jpg']]
my_list = [['bernadette.jpg','raj.jpg','penny.jpg','howard.jpg']]
for each_pair in my_list:
    q = []
    p = {}
    for i, each_person in enumerate(each_pair):
        p[i] = each_pair[i]
        print("Person : ", p[i])
        p[i] = df_matrix.loc[df_matrix['person'] == p[i]]['emb'].values[0]
        p[i] = ast.literal_eval(p[i])
        p[i] = numpy.array(p[i], dtype=float)
        q.append(p[i])
    print(len(q))
    q = numpy.array(q)
    q = q.astype('float32')

    print("Search : ",search(q,"interval"))
