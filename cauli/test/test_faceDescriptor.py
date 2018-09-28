
from cauli.FaceDescriptor import *
import os
import pandas as pd

y=[]
file_name = "/home/buddha/thesis/cauli/test/person_embeddings.csv"
j = 0
df_matrix = pd.read_csv(file_name, sep='\t')
for i in os.listdir("/home/buddha/Desktop/photos/finals/"):
    q = FaceDescriptor('/home/buddha/Desktop/photos/finals/'+i).getDescriptor()
    y.append(q)
    q = q.astype('float32')

    df_len = len(df_matrix)
    df_matrix = df_matrix.append({'person': i, 'emb': []}, ignore_index=True)
    df_matrix.at[df_len, 'emb'] = q.tolist()

with open(file_name, 'a') as f:
    df_matrix.to_csv(file_name, sep = '\t', index= False)