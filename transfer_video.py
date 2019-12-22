import cv2
import numpy as np
import os
import time
import sys
import face_recognition

from utils import Frame_rate_calculator

def main():
    op_dir = 'F:/openpose'
    sys.path.append(op_dir + '/build/python/openpose/Release')
    os.environ['PATH']  = os.environ['PATH'] + ';' + op_dir + '/build/x64/Release;' + op_dir + '/openpose/build/bin;'
    import pyopenpose as op
 
    params = {'model_folder':op_dir + '/models','face':True,'face_detector':2,'body':0}

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    file_dir = 'videos'
    file_path = file_dir + '/20190214_Group1-1_cropped.mp4'
    cap = cv2.VideoCapture(file_path)
    fc = Frame_rate_calculator()

    fc.start_record()
    ret,frame = cap.read()
    while ret:
        
        datum = op.Datum()
        datum.cvInputData = frame
        time1=time.time()
        face_locations = face_recognition.face_locations(frame,model='cnn')
        print(time.time()-time1)
        faceRectangles = [op.Rectangle(location[3],location[0],max(location[1]-location[3],location[2]-location[0]),max(location[1]-location[3],location[2]-location[0])) for location in face_locations]
        datum.faceRectangles = faceRectangles
        
        opWrapper.emplaceAndPop([datum])

        # # detect
        img = np.copy(datum.cvOutputData)

        cv2.putText(img,'FPS:'+str(fc.get_frame_rate()),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        
        fc.frame_end()
        cv2.imshow('i', img)
        
        if cv2.waitKey(1) == 27:
            exit(0)
        
        ret,frame = cap.read()


if __name__ == "__main__":
    main()