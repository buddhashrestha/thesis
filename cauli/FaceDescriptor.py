import dlib
import cv2
import numpy
import os

p = "/home/buddha/thesis/dlib-models/shape_predictor_68_face_landmarks.dat"
landmarks="/home/buddha/thesis/dlib-models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


facerec = dlib.face_recognition_model_v1("/home/buddha/thesis/dlib-models/dlib_face_recognition_resnet_model_v1.dat")

class FaceDescriptor(object):

    def __init__(self, image_path):
        self.image_path = image_path

    def getDescriptor(self):
        if os.path.isfile(self.image_path):
            #File is present
            print ("Yaa hoo")
            image = cv2.imread(self.image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for (i, rect) in enumerate(rects):
                landmark = dlib.shape_predictor(landmarks)(image, rect)
                face_descriptor = facerec.compute_face_descriptor(image, landmark)
            return face_descriptor
        # Return this string if file is not present.
        else:
            return "File not found"



face = FaceDescriptor("/home/buddha/Desktop/buddha.jpg")
print(numpy.asarray(face.getDescriptor()))