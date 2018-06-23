import numpy
import pandas as pd
from .utils.segments import Segments


#this comes from p1 and p2
q1 = numpy.array([-0.03957,0.0159,0.03784,0.03178,-0.10857,-0.04912,-0.04446,-0.08875,0.08702,-0.11579,0.16824,-0.05099,-0.23168,-0.01434,0.02111,0.08825,-0.10974,-0.08089,-0.21508,-0.10312,-0.04391,0.00396,0.01895,-0.00185,-0.12711,-0.32728,-0.13186,-0.08027,0.05784,-0.02188,0.01098,0.03189,-0.19278,-0.07031,0.04278,0.03166,-0.09885,-0.06266,0.17855,-0.03928,-0.21755,0.00416,0.05938,0.21247,0.20348,0.02729,-0.00811,-0.12766,0.09831,-0.19363,-0.01256,0.20605,0.12041,0.09121,0.10923,-0.07219,0.05957,0.14222,-0.14314,0.02365,0.01687,-0.08303,-0.02197,-0.10252,0.13195,0.04069,-0.09316,-0.11487,0.1817,-0.1947,0.00392,0.11932,-0.0956,-0.24175,-0.30322,-0.01014,0.40642,0.18332,-0.17241,-0.02651,-0.06117,-0.09088,0.04655,0.09445,-0.09488,-0.04664,-0.06536,0.09811,0.20961,-0.09043,0.04676,0.1956,0.01257,-0.01395,-0.0025,0.04921,-0.17412,-0.03817,-0.14299,-0.039,0.11551,-0.0773,0.03997,0.09952,-0.12261,0.15803,-0.00143,-0.06661,0.0284,-0.03429,-0.05655,-0.04995,0.20216,-0.17292,0.15109,0.17811,-0.0251,0.09397,0.05758,0.084,-0.09724,-0.05426,-0.1932,-0.07778,0.00567,-0.02787,0.01301,-0.04086
])

q2 = numpy.array([-0.03957,0.0159,0.03784,0.03178,-0.10857,-0.04912,-0.04446,-0.08875,0.08702,-0.11579,0.16824,-0.05099,-0.23168,-0.01434,0.02111,0.08825,-0.10974,-0.08089,-0.21508,-0.10312,-0.04391,0.00396,0.01895,-0.00185,-0.12711,-0.32728,-0.13186,-0.08027,0.05784,-0.02188,0.01098,0.03189,-0.19278,-0.07031,0.04278,0.03166,-0.09885,-0.06266,0.17855,-0.03928,-0.21755,0.00416,0.05938,0.21247,0.20348,0.02729,-0.00811,-0.12766,0.09831,-0.19363,-0.01256,0.20605,0.12041,0.09121,0.10923,-0.07219,0.05957,0.14222,-0.14314,0.02365,0.01687,-0.08303,-0.02197,-0.10252,0.13195,0.04069,-0.09316,-0.11487,0.1817,-0.1947,0.00392,0.11932,-0.0956,-0.24175,-0.30322,-0.01014,0.40642,0.18332,-0.17241,-0.02651,-0.06117,-0.09088,0.04655,0.09445,-0.09488,-0.04664,-0.06536,0.09811,0.20961,-0.09043,0.04676,0.1956,0.01257,-0.01395,-0.0025,0.04921,-0.17412,-0.03817,-0.14299,-0.039,0.11551,-0.0773,0.03997,0.09952,-0.12261,0.15803,-0.00143,-0.06661,0.0284,-0.03429,-0.05655,-0.04995,0.20216,-0.17292,0.15109,0.17811,-0.0251,0.09397,0.05758,0.084,-0.09724,-0.05426,-0.1932,-0.07778,0.00567,-0.02787,0.01301,-0.04086
])



for each_video in bm:
    df_embeddings = pd.read_csv(each_video + '/person_embeddings_mapping.csv',sep='\t')
    #this should be iterable according to query
    person_1 = df_embeddings.loc[df_embeddings['embeddings'] == q1]['person_label']
    person_2 = df_embeddings.loc[df_embeddings['embeddings'] == q2]['person_label']

    df_person_bitmap = pd.read_csv(each_video + '/person_bitmap_vector.csv',sep='\t')
    person_1_bitmap = df_person_bitmap.loc[df_person_bitmap['person_label'] == person_1]['bitmap']
    person_2_bitmap = df_person_bitmap.loc[df_person_bitmap['person_label'] == person_2]['bitmap']

    #then apply.. bitwise 'and' operation.
    vector = "".join([str(int(a*b)) for a,b in zip(person_1_bitmap,person_2_bitmap)])

    from bitmap import BitMap
    bm = BitMap.fromstring(vector)
    print(bm.nonzero())

    #calculate segment now
    c = Segments()
    c.find_continous_segments(bm)
    #yield c
