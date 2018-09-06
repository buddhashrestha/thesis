import numpy
import pandas as pd
import faiss
import ast
import os
import io



def cluster_and_save(embeddings_file,vid_num):

    import os
    import sys
    sys.path.append('../')
    from pyannoteVideo.pyannote.video.face.clustering import FaceClustering
    import pandas as pd
    import numpy as np

    from utils import file

    clustering = FaceClustering(threshold=0.6)
    face_tracks, embeddings = clustering.model.preprocess(embeddings_file)
    #
    result = clustering(face_tracks, features=embeddings)
    #
    current_directory = os.getcwd()
    #
    #
    #video duration in seconds
    video_duration = int(embeddings['time'].iloc[-1])
    print("vid duration:",video_duration)
    ar = [0] * (video_duration + 2)
    embeddings_labels_mappings = {}
    person = {}

    data_matrix = current_directory + "/data/person_to_video_matrix.csv"
    if (os.path.exists(data_matrix)):
        df_matrix = pd.read_csv(data_matrix, sep='\t')
        df_matrix = df_matrix.loc[:, ~df_matrix.columns.str.contains('^Unnamed')]
        print("DF MATRIX before: ", df_matrix)
        cols = list(df_matrix)
        print("Cols before: ", cols)
        cols.insert(0, cols.pop(cols.index('person')))
        df_matrix = df_matrix.ix[:, cols]

        x = str(df_matrix['person'].tolist()).replace("\'", "")
        x = ast.literal_eval(x)
        y = numpy.array(x)
        y = y.astype('float32')
        # now you can train x as you did t
        print("Cols: ", cols)
        video_num = vid_num
        # by default every video is zero
        df_matrix[video_num] = 0

        # searching section

        nlist = 1
        k = 1
        quantizer = faiss.IndexFlatL2(128)  # the other index
        index = faiss.IndexIVFFlat(quantizer, 128, nlist, faiss.METRIC_L2)

        index.train(y)  # t and y ma k farak cha?

        index.add(y)  # add may be a bit slower as well



    for each_label in result.labels():
        if (os.path.exists(data_matrix)):
            p = embeddings.loc[embeddings['track'] == each_label].iloc[:, 2:].values
            r = []
            for i, t in enumerate(p):
                temp = np.array(list(t))
                temp = temp.astype('float32')
                temp = temp.reshape(1, 128)
                r.append(temp)

            dist = -1
            min = 1000
            real_pos = 0
            for j in r:
                D, I = index.search(j, 1)  # actual search
                # if face is not present: then add to the list
                if not I == [[]]:
                    pos = I[0][0]
                    z = y[pos]
                    dist = numpy.linalg.norm(z - j)
                    if dist < min:
                        real_pos = pos
                        min = dist
            print("Min:", min)
            if dist == -1:
                print("theres no person there!!")
                embeddings_labels_mappings[each_label] = list(embeddings.loc[embeddings['track'] == each_label].iloc[0, 2:])

            else:
                if min < 0.578:
                    embeddings_labels_mappings[each_label] = df_matrix.iloc[real_pos, df_matrix.columns.get_loc('person')]
                else:
                    embeddings_labels_mappings[each_label] = list(embeddings.loc[embeddings['track'] == each_label].iloc[0, 2:])

        else:
            embeddings_labels_mappings[each_label] = list(embeddings.loc[embeddings['track'] == each_label].iloc[0, 2:])
        temp = result.itersegments()
        for each_segment in result.itersegments():
            if each_label == result.get_labels(each_segment).pop():
                for i in range(int(each_segment._get_start()),int(each_segment._get_end())):
                    try:
                        ar[i] = 1
                    except:
                        print("I :", i)

        person[each_label] = ar
        ar = [0] * video_duration

    df_person = pd.DataFrame(list(person.items()), columns=['person_label', 'BitMap'])
    df_embeddings = pd.DataFrame(list(embeddings_labels_mappings.items()), columns=['person_label', 'Embeddings'])

    # vid_num = 2
    # print("Current directory: ",current_directory + "/data/")
    # folder_lists = [name for name in os.listdir(current_directory + "/data/" ) if os.path.isdir(current_directory + "/data/"+name)]
    # if folder_lists == []:
    #     vid_num = 1
    # else:
    #     print("folder list: ",folder_lists)
    #     vid_num = int(max(folder_lists)) + 1
    print("vid_num :",vid_num)

    file.check_directory(current_directory + "/data/" + str(vid_num))

    file_name = current_directory + "/data/" + str(vid_num) + '/person_bitmap_vector.csv'
    with open(file_name, 'a') as f:
        df_person.to_csv(file_name, sep = '\t', index= False)

    file_name = current_directory + "/data/" + str(vid_num) + '/person_embeddings_mapping.csv'
    with open(file_name, 'a') as f:
        df_embeddings.to_csv(file_name, sep = '\t', index= False)





    file_name = current_directory + "/data/" + str(vid_num) + '/person_embeddings_mapping.csv'
    data_matrix = current_directory + "/data/person_to_video_matrix.csv"
    if not(os.path.exists(data_matrix)):
        df_matrix = pd.DataFrame(columns=['person'])
        # df_matrix[0] = 0
        video_num = 1
        for each_label in result.labels():
            q = np.array(list(embeddings.loc[embeddings['track'] == each_label].iloc[0, 2:]))
            q = q.astype('float32')
            q = q.reshape(1, 128)
            # if face is not present: then add to the list
            df2 = pd.DataFrame({'person': q.tolist(), video_num: 1})
            df_matrix = pd.concat([df_matrix, df2])
    else:
        d = 128
        df = pd.read_csv(file_name, sep='\t')
        df_matrix = pd.read_csv(data_matrix,sep='\t')
        df_matrix = df_matrix.loc[:, ~df_matrix.columns.str.contains('^Unnamed')]
        print("DF MATRIX before: ", df_matrix)
        cols = list(df_matrix)
        print("Cols before: ",cols)
        cols.insert(0, cols.pop(cols.index('person')))
        df_matrix = df_matrix.ix[:, cols]

        x = str(df_matrix['person'].tolist()).replace("\'","")
        x = ast.literal_eval(x)
        y = numpy.array(x)
        y = y.astype('float32')
        #now you can train x as you did t
        print("Cols: ",cols)
        video_num = vid_num
        #by default every video is zero
        df_matrix[video_num] = 0

        #searching section

        nlist = 1
        k = 1
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)


        index.train(y)# t and y ma k farak cha?

        index.add(y)                  # add may be a bit slower as well

        for each_label in result.labels():
            # print(embeddings)
            # q = np.array(list(embeddings.loc[embeddings['track'] == each_label].iloc[0,2:]))
            # print("q is :",q)
            # print(q.shape)
            # q = q.astype('float32')
            # q = q.reshape(1, 128)
            p = embeddings.loc[embeddings['track'] == each_label].iloc[:,2:].values
            # p = p.astype('float32')
            # print("P:",p.astype('float32'))
            # r=numpy.zeros((len(p),128))
            r=[]
            for i,t in enumerate(p):
                temp = np.array(list(t))
                temp = temp.astype('float32')
                temp = temp.reshape(1,128)
                # print("Temp :",)
                # r[i] = temp
                r.append(temp)
            # v = r[0]
            # print("v : ",r.shape)
            # exit(0)
            D, I = index.search(r[0], k)     # actual search
            print("I: ",I)


            dist = -1
            min = 1000
            real_pos = 0
            for j in r:
                D, I = index.search(j, k)  # actual search
            #if face is not present: then add to the list
                if not I == [[]]:
                    pos = I[0][0]
                    z = y[pos]
                    dist = numpy.linalg.norm(z - j)
                    if dist<min:
                        real_pos = pos
                        min = dist
            print("Min:", min)
            if dist == -1:
                print("theres no person there!!")
                df2 = pd.DataFrame({'person': r[0].tolist(), video_num: 1})
                df_matrix = pd.concat([df_matrix, df2])

            else:
                if min < 0.578:
                    df_matrix.iloc[real_pos, df_matrix.columns.get_loc(video_num)] = 1
                else:
                    df2 = pd.DataFrame({'person': r[0].tolist(), video_num: 1})
                    df_matrix = pd.concat([df_matrix, df2])

            # if I ==[[]]:
            #     print("theres no person there!!")
            #     df2 = pd.DataFrame({'person':q.tolist(),video_num:1})
            #     df_matrix = pd.concat([df_matrix,df2])
            # else:
            #     print("there is that person")
            #     pos = I[0][0]
            #     z = y[pos]
            #     print('person vector :', z)
            #     dist = numpy.linalg.norm(z - q)
            #     if dist < 0.578:
            #         df_matrix.iloc[pos, df_matrix.columns.get_loc(video_num)] = 1
            #     else:
            #         df2 = pd.DataFrame({'person': q.tolist(), video_num: 1})
            #         df_matrix = pd.concat([df_matrix, df2])
    cols = list(df_matrix)
    cols.insert(0, cols.pop(cols.index('person')))
    df_matrix = df_matrix.ix[:, cols]
    print("DF_matrix after: ",df_matrix)
    file_name = current_directory + '/data/person_to_video_matrix.csv'

    with open(file_name, 'a') as f:
        df_matrix.to_csv(data_matrix, sep = '\t', index= False)

cluster_and_save("./data/"+ str(2) + "/"+  "friends1_720.embedding.txt", 2)
#