import os
from argparse import ArgumentParser
from subprocess import call
directory = "/home/buddha/thesis/cauli/data/1/"
call(["python","/home/buddha/thesis/pyannoteVideo/scripts/pyannote-face.py", "demo",
                       "/home/buddha/thesis/pyannote-data/" + "TheBigBangTheory.mkv",
                       directory + "TheBigBangTheory.track.txt",
                       "/home/buddha/thesis/pyannote-data/" + "demo.track.mp4"])

print("done with demo.")