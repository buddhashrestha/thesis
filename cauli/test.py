from pyannoteVideo.pyannote.video.face.clustering import FaceClustering
import numpy as np
clustering = FaceClustering(threshold=0.6)
face_tracks, embeddings = clustering.model.preprocess('/home/buddha/thesis/pyannote-data/TheBigBangTheory.embedding.txt')

result = clustering(face_tracks, features=embeddings)
#video duration in seconds
video_duration = int(embeddings['time'].iloc[-1])
print(type(result))
ar = [0] * video_duration
print(ar)
person = {}
for each_label in result.labels():
    for each_segment in result.itersegments():
        if each_label == result.get_labels(each_segment).pop():
            for i in range(int(each_segment._get_start()),int(each_segment._get_end())):
                ar[i] = 1
    person[each_label] = ar
    ar = [0] * video_duration
print(person)