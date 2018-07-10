import numpy
import pandas as pd
import faiss
from utils.segments import Segments
from findVideos import findVideos
import ast
import json

#this comes from p1 and p2
q1 = numpy.array([-0.09394396,-0.0325009,0.04033981,0.058163,-0.07830061,-0.00957991,0.00967481,-0.01413031,0.09056109,0.01226895,0.26494166,-0.02123076,-0.24777047,-0.01763329,-0.07384738,0.06715217,-0.12217229,-0.08997168,-0.07580122,-0.04950837,0.04795098,0.13332213,-0.01127091,0.02197756,-0.11553577,-0.27555162,-0.07677643,-0.13606225,0.04441723,-0.15569574,0.00895849,0.14969787,-0.11935396,-0.06838058,0.07786268,0.03718393,-0.06408153,-0.02089238,0.23585948,0.0805631,-0.15866175,0.01856041,0.00116307,0.41145056,0.16814063,0.02847197,-0.03902237,-0.05839182,0.07346018,-0.2937974,0.11478464,0.17386746,0.17906441,0.07801438,0.06278698,-0.19769681,-0.02882751,0.14530656,-0.1747098,0.18388774,0.02294914,-0.13189943,0.02977884,-0.12073777,0.12760715,0.01692488,-0.10679697,-0.05800077,0.18581267,-0.10150822,-0.05918678,0.1363336,-0.13339984,-0.26164994,-0.22236374,0.06707402,0.3977724,0.11571225,-0.23185259,0.0054906,-0.01040609,-0.03162199,0.04665727,0.08441081,-0.05217192,-0.03122335,-0.07699484,0.00657031,0.17925991,-0.00695905,-0.06547561,0.22437108,-0.0341662,-0.03394335,0.13902065,0.03442689,-0.00662828,-0.03761093,-0.1410363,-0.02334402,-0.00892982,-0.22861557,-0.01857813,0.06306355,-0.14851157,0.18434337,0.02413726,-0.01379691,-0.00887952,-0.06088896,-0.07728448,0.06671096,0.24838527,-0.2970275,0.25738838,0.14351685,0.08496048,0.20006748,-0.00258899,0.0801791,-0.06668909,-0.06449306,-0.18159339,-0.07939189,0.05472237,-0.00587219,0.02267785,0.03292688])

q2 = numpy.array([-0.13995799,0.07357684,0.06817903,-0.03358017,-0.14017977,0.02780785,-0.00525929,-0.01784202,0.06178749,-0.10179561,0.18950391,-0.01475109,-0.27838451,-0.0266105,-0.08373968,0.09460582,-0.1788917,-0.17850886,-0.12365769,-0.14781366,0.03512874,0.00903247,-0.02803879,0.03783317,-0.16920707,-0.26304603,-0.06992229,-0.10061786,0.05101402,-0.16632022,0.11655138,0.08433184,-0.12079179,0.03596556,-0.0121152,0.12947829,0.04827371,-0.08188809,0.18349151,-0.00281307,-0.15602723,0.06405699,0.04490693,0.31951201,0.17494193,0.01719636,-0.0419702,-0.08516614,0.12460594,-0.21499792,0.06585026,0.19651721,0.09745228,0.03903785,0.10535515,-0.13724521,0.08984233,0.13477094,-0.21704961,0.05335507,0.0309218,-0.05781218,0.0019028,-0.09391791,0.0646577,0.06179845,-0.08008289,-0.14667159,0.20857613,-0.18729421,-0.06652444,0.14170116,-0.1157265,-0.1452689,-0.24668945,0.04148631,0.40796083,0.21065351,-0.11024678,0.02329403,-0.01277202,-0.04557295,0.07328524,0.03643153,-0.17143802,-0.08823329,-0.10314541,0.11035879,0.15545008,0.09105477,-0.0749627,0.19225572,0.03504357,-0.04319753,0.07131915,0.02174823,-0.08037397,0.02350202,-0.12592137,-0.0058411,0.06064202,-0.16452999,-0.01263573,0.0567708,-0.13627924,0.06755719,-0.01812054,-0.00742016,-0.0428388,0.01858514,-0.18413505,0.08322754,0.24775918,-0.29474801,0.18072528,0.21811365,0.03798047,0.06725502,0.17808905,0.0480503,-0.01554211,-0.12508231,-0.1766559,-0.07591728,0.04641546,-0.08250329,0.07233171,0.03478055])

videos = findVideos()

for each_video in videos:
    df_embeddings = pd.read_csv(str(each_video) + '/person_embeddings_mapping.csv',sep='\t')
    cols = list(df_embeddings)
    print(cols)
    cols = list(df_embeddings)
    cols.insert(0, cols.pop(cols.index('Embeddings')))
    df_embeddings = df_embeddings.ix[:, cols]

    x = str(df_embeddings['Embeddings'].tolist()).replace("\'", "")
    x = ast.literal_eval(x)
    y = numpy.array(x)
    print(y.shape)
    y = y.astype('float32')
    d = 128
    nlist = 1
    k = 1
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    q = []

    index.train(y)  # t and y ma k farak cha?
    q.append(q1)
    q.append(q2)

    q = numpy.array(q)
    q = q.astype('float32')

    index.add(y)  # add may be a bit slower as well

    D, I = index.search(q, k)  # actual search
    print(I)  # neighbors of the 5 last queries

    # if face is not present: then add to the list
    if I == [[]]:
        print("Not found")
    else:
        pos1 = I[0][0]
        pos2 = I[1][0]
        p1 = df_embeddings.iloc[pos1]  # corresponding to q1 and q2
        p2 = df_embeddings.iloc[pos2]  # for d2
        p1 = list(df_embeddings.iloc[pos1, 2:])
        p2 = list(df_embeddings.iloc[pos2, 2:])
        print("p1:",p1[0])
        print("p1:",p2[0])

    #this should be iterable according to query
    ##similarity search!!
    # person_1 = df_embeddings.loc[df_embeddings['Embeddings'] == q1]['person_label']
    # person_2 = df_embeddings.loc[df_embeddings['embeddings'] == q2]['person_label']
    #
    df_person_bitmap = pd.read_csv(str(each_video) +'/person_bitmap_vector.csv',sep='\t')
    print(df_person_bitmap['person_label'])

    person_1_bitmap = df_person_bitmap.loc[df_person_bitmap['person_label'] == int(p1[0])]['BitMap'].values[0]
    person_2_bitmap = df_person_bitmap.loc[df_person_bitmap['person_label'] == int(p2[0])]['BitMap'].values[0]
    #then apply.. bitwise 'and' operation.
    person_1_bitmap = str(person_1_bitmap).replace("\'", "")
    person_2_bitmap = str(person_2_bitmap).replace("\'", "")

    person_1_bitmap = json.loads(person_1_bitmap)
    person_2_bitmap = json.loads(person_2_bitmap)

    person_1_bitmap = numpy.array(person_1_bitmap)
    person_2_bitmap = numpy.array(person_2_bitmap)

    print("p1:",person_1_bitmap)
    print("p2:",person_2_bitmap)

    # person_2_bitmap = [int(numeric_string) for numeric_string in person_1_bitmap]
    vector = "".join([str(int( a*b)) for a,b in zip(person_1_bitmap,person_2_bitmap)])
    print("Vector:",vector)
    from bitmap.src.bitmap import BitMap
    bm = BitMap.fromstring(vector)
    x = numpy.array(list(bm.nonzero()))
    x = len(vector)-x
    x = x[::-1]
    print("x: ",x)
    #calculate segment now
    c = Segments()
    print("Contigous: ",c.find_continous_segments(x))
    #yield c


#TODO:loadup new video, run a person search on a custom photo using FaceDescriptor