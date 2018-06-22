from imutils import face_utils
import dlib
import cv2
import numpy
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
from pyannote.video import Face

p = "/home/buddha/thesis/dlib-models/shape_predictor_68_face_landmarks.dat"
landmarks="/home/buddha/thesis/dlib-models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# load the input image and convert it to grayscale
# image = cv2.imread("/home/buddha/Desktop/buddha.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image

facerec = dlib.face_recognition_model_v1("/home/buddha/thesis/dlib-models/dlib_face_recognition_resnet_model_v1.dat")

class FaceDescriptor(object):
    def __init__(self, image):
        self.image = cv2.imread(image)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.rects = detector(self.gray, 0)

    def getDescriptor(self):

        # loop over the face detections
        for (i, rect) in enumerate(self.rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            # shape = predictor(gray, rect)
            # shape = face_utils.shape_to_np(shape)
            landmark = dlib.shape_predictor(landmarks)(self.image, rect)
            face_descriptor = facerec.compute_face_descriptor(self.image, landmark)
        return face_descriptor

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
#     for (x, y) in shape:
#         cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
#
# # show the output image with the face detections + facial landmarks
# cv2.imshow("Output", image)
# cv2.waitKey(0)

from collections import namedtuple
Range = namedtuple("Range", ["start", "end"])

face = FaceDescriptor("/home/buddha/Desktop/buddha.jpg")
print(numpy.asarray(face.getDescriptor()))