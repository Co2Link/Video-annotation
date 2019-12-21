import sys
import cv2
import os
from sys import platform
import argparse
import time

try:
    sys.path.append('F:/local/openpose/build/python/openpose/Release')
    os.environ['PATH']  = os.environ['PATH'] + ';' + 'F:/local/openpose/build/x64/Release;' + 'F:/local/openpose/build/bin;'
    import pyopenpose as op

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = {'model_folder':'F:/local/openpose/models','face':True,'face_detector':2,'body':0}

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read image and face rectangle locations
    image_path = 'images/2.jpg'
    imageToProcess = cv2.imread(image_path)
    faceRectangles = [
        op.Rectangle(330.119385, 277.532715, 48.717274, 48.717274),
        op.Rectangle(24.036991, 267.918793, 65.175171, 65.175171),
        op.Rectangle(151.803436, 32.477852, 108.295761, 108.295761),
    ]

    # Create new datum
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    datum.faceRectangles = faceRectangles

    # Process and display image
    opWrapper.emplaceAndPop([datum])
    print("Face keypoints: \n" + str(datum.faceKeypoints))
    cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)