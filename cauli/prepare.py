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

# exit(0)
directory = "/home/buddha/thesis/cauli/data/"+ str(vid_num) +"/"
if not os.path.exists(directory):
    os.makedirs(directory)

file_name = args.filename
movie_name = file_name.rsplit(".")[0]
shots_name = movie_name + ".shots.json"
track_name = movie_name + ".track.txt"
landmark_name = movie_name + ".landmarks.txt"
embeddings = movie_name + ".embedding.txt"
demo = movie_name+ ".track.mp4"

call(["python", "/home/buddha/thesis/pyannoteVideo/scripts/pyannote-structure.py",
      "shot","--verbose","/home/buddha/thesis/pyannote-data/" + file_name,
      directory+ shots_name])
print("done with shots.")

call(["python", "/home/buddha/thesis/pyannoteVideo/scripts/pyannote-face.py",
      "track","--verbose","--every=0.5","/home/buddha/thesis/pyannote-data/" + file_name,
      directory + shots_name,
      directory + track_name])
print("done with track.")

call(["python","/home/buddha/thesis/pyannoteVideo/scripts/pyannote-face.py", "demo",
                       "/home/buddha/thesis/pyannote-data/" + file_name,
                       directory + track_name,
                       directory + demo])

print("done with demo.")

call(["python", "../pyannoteVideo/scripts/pyannote-face.py",
      "extract","--verbose","../pyannote-data/" + file_name,
      "./data/" + str(vid_num) + "/" + track_name,
      "../dlib-models/shape_predictor_68_face_landmarks.dat",
      "../dlib-models/dlib_face_recognition_resnet_model_v1.dat",
      "./data/" + str(vid_num) + "/" + landmark_name,
      "./data/" + str(vid_num) + "/" +  embeddings])
print("Done with embeddings.")

from cauli.person_to_bitmap_vector import *

# cluster_and_save("./data/"+ str(vid_num) + "/"+  embeddings, vid_num)

print("done with everything")
print("finished..")

#to run : save the video to pyannote-data folder, then run : python prepare.py -f "TheBigBangTheory.mkv"