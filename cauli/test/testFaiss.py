import numpy
import pandas as pd
import faiss
import ast
import os

d = 128
# df = pd.read_csv('/home/buddha/thesis/cauli/data/person_to_video_matrix.csv', sep='\t')
# df_matrix = pd.read_csv('/home/buddha/thesis/cauli/data/person_to_video_matrix.csv',sep='\t')
# df_matrix = df_matrix.loc[:, ~df_matrix.columns.str.contains('^Unnamed')]
# print("DF MATRIX before: ", df_matrix)
# cols = list(df_matrix)
# print("Cols before: ",cols)
# cols.insert(0, cols.pop(cols.index('person')))
# df_matrix = df_matrix.ix[:, cols]
#
# x = str(df_matrix['person'].tolist()).replace("\'","")
# x = ast.literal_eval(x)
# y = numpy.array(x)
# print(type(y))
# y = y.astype('float32')
# #now you can train x as you did t
# print("Cols: ",cols)
# video_num = 1
# #by default every video is zero
# df_matrix[video_num] = 0

y = []
import os
from cauli.FaceDescriptor import *
#TODO: do it if the file doesnot exists!!
file_name = "/home/buddha/thesis/cauli/data/person_to_video_matrix.csv"
j = 0
df_matrix = pd.read_csv(file_name, sep='\t')
df_matrix = df_matrix.loc[:, ~df_matrix.columns.str.contains('^Unnamed')]
cols = list(df_matrix)
cols.insert(0, cols.pop(cols.index('person')))
df_matrix = df_matrix.ix[:, cols]

# for i in os.listdir("/home/buddha/Desktop/photos/finals/"):
#     q = FaceDescriptor('/home/buddha/Desktop/photos/finals/'+i).getDescriptor()
#     y.append(q)
#
#     # df_matrix[0] = 0
#     video_num = 1
#
#     q = q.astype('float32')
#     q = q.reshape(1, 128)
#     # if face is not present: then add to the list
#     df2 = pd.DataFrame({'person': q.tolist()})
#     df_matrix = pd.concat([df_matrix, df2])
#     j= j +1
# #
# print(df_matrix)
# with open(file_name, 'a') as f:
#     df_matrix.to_csv(file_name, sep = '\t', index= False)
# exit(0)

y = numpy.array(y)

y = y.astype('float32')
print(y)
#searching section

nlist = 1
k = 4
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)


index.train(y)# t and y ma k farak cha?

index.add(y)                  # add may be a bit slower as well

leonard = [-0.13995799, 0.07357684, 0.06817903, -0.03358017, -0.14017977, 0.02780785, -0.00525929, -0.01784202,
           0.06178749, -0.10179561, 0.18950391, -0.01475109, -0.27838451, -0.0266105, -0.08373968, 0.09460582,
           -0.1788917, -0.17850886, -0.12365769, -0.14781366, 0.03512874, 0.00903247, -0.02803879, 0.03783317,
           -0.16920707, -0.26304603, -0.06992229, -0.10061786, 0.05101402, -0.16632022, 0.11655138, 0.08433184,
           -0.12079179, 0.03596556, -0.0121152, 0.12947829, 0.04827371, -0.08188809, 0.18349151, -0.00281307,
           -0.15602723, 0.06405699, 0.04490693, 0.31951201, 0.17494193, 0.01719636, -0.0419702, -0.08516614,
           0.12460594, -0.21499792, 0.06585026, 0.19651721, 0.09745228, 0.03903785, 0.10535515, -0.13724521,
           0.08984233, 0.13477094, -0.21704961, 0.05335507, 0.0309218, -0.05781218, 0.0019028, -0.09391791,
           0.0646577, 0.06179845, -0.08008289, -0.14667159, 0.20857613, -0.18729421, -0.06652444, 0.14170116,
           -0.1157265, -0.1452689, -0.24668945, 0.04148631, 0.40796083, 0.21065351, -0.11024678, 0.02329403,
           -0.01277202, -0.04557295, 0.07328524, 0.03643153, -0.17143802, -0.08823329, -0.10314541, 0.11035879,
           0.15545008, 0.09105477, -0.0749627, 0.19225572, 0.03504357, -0.04319753, 0.07131915, 0.02174823,
           -0.08037397, 0.02350202, -0.12592137, -0.0058411, 0.06064202, -0.16452999, -0.01263573, 0.0567708,
           -0.13627924, 0.06755719, -0.01812054, -0.00742016, -0.0428388, 0.01858514, -0.18413505, 0.08322754,
           0.24775918, -0.29474801, 0.18072528, 0.21811365, 0.03798047, 0.06725502, 0.17808905, 0.0480503,
           -0.01554211, -0.12508231, -0.1766559, -0.07591728, 0.04641546, -0.08250329, 0.07233171, 0.03478055]

qq = []
qq.append(leonard)


qq = numpy.array(qq)
qq = qq.astype('float32')

D, I = index.search(qq, 1)     # actual search
print("I: ",I)
#if face is not present: then add to the list
if I ==[[]]:
    print("theres no person there!!")
    # df2 = pd.DataFrame({'person':q.tolist(),video_num:1})
    # df_matrix = pd.concat([df_matrix,df2])
else:
    print("there is that person")
    pos = I[0][0]
    z = y[pos]
    print('person vector :', z)
    dist = numpy.linalg.norm(z - leonard)
    print('dist :',dist)
    # df_matrix.iloc[pos, df_matrix.columns.get_loc(video_num)] = 1


'''
sheldon = [-0.09394396, -0.0325009, 0.04033981, 0.058163, -0.07830061, -0.00957991, 0.00967481, -0.01413031,
           0.09056109, 0.01226895, 0.26494166, -0.02123076, -0.24777047, -0.01763329, -0.07384738, 0.06715217,
           -0.12217229, -0.08997168, -0.07580122, -0.04950837, 0.04795098, 0.13332213, -0.01127091, 0.02197756,
           -0.11553577, -0.27555162, -0.07677643, -0.13606225, 0.04441723, -0.15569574, 0.00895849, 0.14969787,
           -0.11935396, -0.06838058, 0.07786268, 0.03718393, -0.06408153, -0.02089238, 0.23585948, 0.0805631,
           -0.15866175, 0.01856041, 0.00116307, 0.41145056, 0.16814063, 0.02847197, -0.03902237, -0.05839182,
           0.07346018, -0.2937974, 0.11478464, 0.17386746, 0.17906441, 0.07801438, 0.06278698, -0.19769681,
           -0.02882751, 0.14530656, -0.1747098, 0.18388774, 0.02294914, -0.13189943, 0.02977884, -0.12073777,
           0.12760715, 0.01692488, -0.10679697, -0.05800077, 0.18581267, -0.10150822, -0.05918678, 0.1363336,
           -0.13339984, -0.26164994, -0.22236374, 0.06707402, 0.3977724, 0.11571225, -0.23185259, 0.0054906,
           -0.01040609, -0.03162199, 0.04665727, 0.08441081, -0.05217192, -0.03122335, -0.07699484, 0.00657031,
           0.17925991, -0.00695905, -0.06547561, 0.22437108, -0.0341662, -0.03394335, 0.13902065, 0.03442689,
           -0.00662828, -0.03761093, -0.1410363, -0.02334402, -0.00892982, -0.22861557, -0.01857813, 0.06306355,
           -0.14851157, 0.18434337, 0.02413726, -0.01379691, -0.00887952, -0.06088896, -0.07728448, 0.06671096,
           0.24838527, -0.2970275, 0.25738838, 0.14351685, 0.08496048, 0.20006748, -0.00258899, 0.0801791,
           -0.06668909, -0.06449306, -0.18159339, -0.07939189, 0.05472237, -0.00587219, 0.02267785, 0.03292688]

leonard = [-0.13995799, 0.07357684, 0.06817903, -0.03358017, -0.14017977, 0.02780785, -0.00525929, -0.01784202,
           0.06178749, -0.10179561, 0.18950391, -0.01475109, -0.27838451, -0.0266105, -0.08373968, 0.09460582,
           -0.1788917, -0.17850886, -0.12365769, -0.14781366, 0.03512874, 0.00903247, -0.02803879, 0.03783317,
           -0.16920707, -0.26304603, -0.06992229, -0.10061786, 0.05101402, -0.16632022, 0.11655138, 0.08433184,
           -0.12079179, 0.03596556, -0.0121152, 0.12947829, 0.04827371, -0.08188809, 0.18349151, -0.00281307,
           -0.15602723, 0.06405699, 0.04490693, 0.31951201, 0.17494193, 0.01719636, -0.0419702, -0.08516614,
           0.12460594, -0.21499792, 0.06585026, 0.19651721, 0.09745228, 0.03903785, 0.10535515, -0.13724521,
           0.08984233, 0.13477094, -0.21704961, 0.05335507, 0.0309218, -0.05781218, 0.0019028, -0.09391791,
           0.0646577, 0.06179845, -0.08008289, -0.14667159, 0.20857613, -0.18729421, -0.06652444, 0.14170116,
           -0.1157265, -0.1452689, -0.24668945, 0.04148631, 0.40796083, 0.21065351, -0.11024678, 0.02329403,
           -0.01277202, -0.04557295, 0.07328524, 0.03643153, -0.17143802, -0.08823329, -0.10314541, 0.11035879,
           0.15545008, 0.09105477, -0.0749627, 0.19225572, 0.03504357, -0.04319753, 0.07131915, 0.02174823,
           -0.08037397, 0.02350202, -0.12592137, -0.0058411, 0.06064202, -0.16452999, -0.01263573, 0.0567708,
           -0.13627924, 0.06755719, -0.01812054, -0.00742016, -0.0428388, 0.01858514, -0.18413505, 0.08322754,
           0.24775918, -0.29474801, 0.18072528, 0.21811365, 0.03798047, 0.06725502, 0.17808905, 0.0480503,
           -0.01554211, -0.12508231, -0.1766559, -0.07591728, 0.04641546, -0.08250329, 0.07233171, 0.03478055]

einstein =  [-1.18590906e-01,5.18187769e-02,5.08944355e-02,-2.96264272e-02
,9.87266097e-03,-2.84665655e-02,5.44366287e-03,-6.72525764e-02
,1.74365535e-01,-1.13969222e-01,1.31720066e-01,1.71529353e-02
,-2.01174751e-01,-5.73378503e-02,1.66792795e-02,9.84659865e-02
,-9.91091058e-02,-8.85752961e-02,-1.70864150e-01,-9.15386379e-02
,-4.92944866e-02,1.06168248e-01,5.85778244e-02,-4.45470214e-02
,-8.54505002e-02,-4.15925175e-01,-8.70231092e-02,-1.00485653e-01
,3.35049555e-02,-9.37458649e-02,4.98125143e-02,3.89686450e-02
,-2.17812911e-01,-3.61120366e-02,2.71756260e-04,8.38618875e-02
,-8.45792070e-02,-2.57666819e-02,1.94614351e-01,1.48445800e-01
,-2.30072170e-01,7.68971518e-02,1.31668970e-02,2.86541820e-01
,2.40368232e-01,1.10332146e-02,-4.09415225e-03,-9.46811959e-02
,1.03274189e-01,-2.19359413e-01,1.12226039e-01,1.34492397e-01
,1.26337126e-01,6.13042600e-02,1.53247505e-01,-6.27315864e-02
,2.10989472e-02,1.15062438e-01,-1.91664696e-01,9.48088914e-02
,9.32547450e-02,-1.08510524e-01,-3.23885195e-02,4.45278129e-03
,1.11162439e-01,1.00667782e-01,-9.65835154e-02,-1.42242074e-01
,1.15158327e-01,-1.98714167e-01,-4.74210791e-02,9.71777886e-02
,-5.95812239e-02,-1.64220706e-01,-3.34909052e-01,8.70205015e-02
,3.53250206e-01,1.55502692e-01,-1.48625523e-01,-1.32036852e-02
,-9.17755067e-02,-5.01982123e-02,9.52153727e-02,1.56939164e-01
,-5.24024591e-02,-1.19406939e-01,-1.09490715e-02,-2.31120288e-02
,1.72749668e-01,6.85811266e-02,-1.19730331e-01,1.48926556e-01
,-6.41097054e-02,-1.90565493e-02,-4.69634645e-02,2.16365866e-02
,-1.81284502e-01,-5.50339483e-02,-7.12295771e-02,-1.08495854e-01
,1.78381486e-03,-8.48575085e-02,-4.25664224e-02,1.31438911e-01
,-2.18736097e-01,2.17922345e-01,-3.35240103e-02,-6.35516420e-02
,9.42047015e-02,2.70053782e-02,-1.65708922e-02,1.59000512e-02
,9.72331017e-02,-2.56779134e-01,2.66865253e-01,2.07905114e-01
,-2.80461926e-02,1.29835606e-01,7.20018819e-02,5.95808998e-02
,-2.87829451e-02,5.57796918e-02,-1.89550996e-01,-1.23968810e-01
,-2.49511972e-02,5.13287410e-02,5.15042767e-02,-8.49819928e-03]

will = [-2.04010773e-03,1.20580025e-01,1.68546409e-04,-6.81897327e-02
,-1.50781080e-01,4.96236980e-02,-5.19941486e-02,-1.34188876e-01
,9.86745954e-02,-1.74888782e-02,2.25662261e-01,-5.17177917e-02
,-2.91598111e-01,-6.36585429e-02,-4.25830707e-02,7.28473067e-02
,-1.46822795e-01,-5.61609678e-03,-1.37674645e-01,-1.29761606e-01
,-2.58173887e-02,1.70792900e-02,7.47252479e-02,-5.77010065e-02
,-9.75931138e-02,-2.68536955e-01,-3.98041308e-02,-1.38793856e-01
,2.40195524e-02,-1.82100877e-01,1.83055289e-02,9.32324305e-03
,-2.71532506e-01,-6.33505881e-02,-1.01820208e-01,-1.11387167e-02
,-1.59791246e-01,-4.89222072e-02,1.77511990e-01,6.71637570e-03
,-8.09674934e-02,6.60340413e-02,6.85856119e-02,2.32192501e-01
,2.53583044e-01,-2.56494544e-02,-1.09092901e-02,-5.12236431e-02
,1.36427507e-01,-2.69395113e-01,2.73092091e-02,1.56728044e-01
,1.13605663e-01,5.84602393e-02,7.60088116e-02,-7.91694745e-02
,6.46653632e-03,1.00987181e-01,-1.70559078e-01,5.49373962e-02
,6.76369742e-02,1.48620382e-02,-9.96125489e-02,-3.62632759e-02
,2.14497685e-01,9.14853662e-02,-6.62541687e-02,-9.94952619e-02
,1.01687059e-01,-1.35046214e-01,-5.95275573e-02,4.60091718e-02
,-1.16482429e-01,-1.87712431e-01,-2.84854203e-01,8.39355588e-02
,3.06227356e-01,6.22549839e-02,-1.78411648e-01,2.79858354e-02
,-1.11546479e-01,-9.19821486e-02,2.92975269e-02,-2.07464695e-02
,-2.62625255e-02,-4.62103598e-02,-8.28014314e-02,7.55291432e-02
,2.06201360e-01,-1.82864055e-01,1.68697536e-02,2.37035438e-01
,3.30011286e-02,-5.33931255e-02,8.97432305e-03,3.27581763e-02
,-1.32524610e-01,-2.23764703e-02,-1.74089462e-01,-3.94859631e-03
,5.86322211e-02,-1.59150824e-01,-6.11823536e-02,4.20063101e-02
,-2.03067914e-01,1.75458759e-01,5.17642051e-02,-9.99006629e-02
,-2.31304206e-02,-1.53720945e-01,-9.43153203e-02,-3.49410716e-03
,2.19476745e-01,-2.61898279e-01,2.72775352e-01,1.53620228e-01
,1.33356377e-02,1.04224473e-01,4.79194261e-02,2.81414986e-02
,-1.35316979e-02,4.61590365e-02,-1.06680952e-01,-1.28902614e-01
,1.43208429e-02,5.99296391e-03,-6.38523772e-02,2.34818906e-02]

jurgen = [-0.07722384,0.03862969,0.11232487,-0.07928048,-0.12589082,-0.06844431
,0.01876052,-0.02569746,0.12161407,-0.01321536,0.12557891,0.07807396
,-0.26364088,-0.05781354,0.04420178,0.02651414,-0.08880096,-0.17571162
,-0.12198005,-0.12226233,-0.05889008,0.01064788,-0.02659107,-0.00814604
,-0.15042995,-0.24161445,-0.08560306,-0.11482064,0.14683066,-0.21525861
,0.04174795,0.04448514,-0.20568976,-0.07468145,-0.009667,0.11196363
,-0.0894431,-0.14120434,0.23966974,0.03678298,-0.17124149,-0.03769939
,0.0307141,0.282462,0.2618497,-0.05116623,0.05773098,-0.07523727
,0.1678014,-0.27086392,0.13853703,0.13747437,0.10369678,0.10628232
,0.12423684,-0.21400465,0.07430235,0.07989737,-0.31081131,0.11694349
,0.09688658,-0.09447797,-0.08675412,-0.06606281,0.13295445,0.04410006
,-0.09748738,-0.14905056,0.14526504,-0.26489416,0.00999036,0.17978071
,-0.05574533,-0.13209142,-0.29793444,0.06785395,0.32676262,0.19514124
,-0.11643145,0.06936578,-0.07798179,-0.09348666,0.05119462,0.04209178
,-0.10289548,0.0488796,-0.06156423,0.08948641,0.18388033,0.05236822
,-0.02731201,0.15984327,0.01775069,-0.00681705,0.06842531,0.05611343
,-0.09509671,-0.06190236,-0.08852672,0.02038449,0.04989311,-0.18726249
,0.01208717,0.03383485,-0.14724502,0.1806177,0.02760591,-0.02794529
,-0.08495249,-0.15567699,-0.14716242,0.04063524,0.23431684,-0.3523261
,0.2479147,0.1689228,-0.01604912,0.0978321,0.08192257,0.02295081
,-0.03542054,-0.09943051,-0.09774172,-0.0295199,-0.00292781,-0.05809616
,0.10721888,0.04816229]

qq = []
qq.append(leonard)


qq = numpy.array(qq)
qq = qq.astype('float32')

current_directory = os.getcwd()
data_matrix = "/home/buddha/thesis/cauli/data/person_to_video_matrix.csv"
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

d = 128
#searching section

nlist = 1
k = 2
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)


index.train(y)# t and y ma k farak cha?

index.add(y)


D, I = index.search(qq, 1)     # actual search
print("I: ",I)
#if face is not present: then add to the list
if I ==[[]]:
    print("theres no person there!!")
    # df2 = pd.DataFrame({'person':q.tolist(),video_num:1})
    # df_matrix = pd.concat([df_matrix,df2])
else:
    print("there is that person")
    pos = I[0][0]
    z = y[pos]
    print('person vector :', z)
    dist = numpy.linalg.norm(z - jurgen)
    print('dist :',dist)
    # df_matrix.iloc[pos, df_matrix.columns.get_loc(video_num)] = 1
'''
