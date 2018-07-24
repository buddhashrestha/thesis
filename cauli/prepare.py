import os
from argparse import ArgumentParser
from subprocess import call

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename",
                    help="write report to FILE", metavar="FILE")

args = parser.parse_args()

print("Processing",args.filename)

current_directory = os.getcwd()
folder_lists = [name for name in os.listdir(current_directory + "/data/" ) if os.path.isdir(current_directory + "/data/"+name)]
if folder_lists == []:
    vid_num = 1
else:
    print("folder list: ",folder_lists)
    vid_num = int(max(folder_lists)) + 1

vid_num = 1

file_name = args.filename
movie_name = file_name.rsplit(".")[0]
shots_name = movie_name + ".shots.json"
track_name = movie_name + ".track.txt"
landmark_name = movie_name + ".landmarks.txt"
embeddings = movie_name + ".embedding.txt"

call(["python", "/home/buddha/thesis/pyannoteVideo/scripts/pyannote-structure.py",
      "shot","--verbose","/home/buddha/thesis/pyannote-data/" + file_name,
      "/home/buddha/thesis/cauli/data/"+ str(vid_num) +"/"+ shots_name])
print("done with shots.")

call(["python", "/home/buddha/thesis/pyannoteVideo/scripts/pyannote-face.py",
      "track","--verbose","--every=0.5","/home/buddha/thesis/pyannote-data/" + file_name,
      "/home/buddha/thesis/cauli/data/"+ str(vid_num) +"/"+ shots_name,
      "/home/buddha/thesis/cauli/data/"+ str(vid_num) +"/"+ track_name])
print("done with track.")


call(["python", "../pyannoteVideo/scripts/pyannote-face.py",
      "extract","--verbose","../pyannote-data/" + file_name,
      "./data/" + str(vid_num) + "/" + track_name,
      "../dlib-models/shape_predictor_68_face_landmarks.dat",
      "../dlib-models/dlib_face_recognition_resnet_model_v1.dat",
      "./data/" + str(vid_num) + "/" + landmark_name,
      "./data/" + str(vid_num) + "/" +  embeddings])
print("Done with embeddings.")

from person_to_bitmap_vector import *

cluster_and_save("./data/"+ str(vid_num) + "/"+  embeddings, vid_num)

print("done with everything")
print("finished..")

#to run : python prepare.py -f "TheBigBangTheory.mkv"