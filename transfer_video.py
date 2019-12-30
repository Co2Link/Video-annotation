import cv2
import numpy as np
import os
import time
import sys
import face_recognition
import pickle
from tqdm import tqdm

from utils import Frame_rate_calculator


def main():
    op_dir = 'F:/local/openpose'
    sys.path.append(op_dir + '/build/python/openpose/Release')
    os.environ['PATH'] = os.environ['PATH'] + ';' + op_dir + \
        '/build/x64/Release;' + op_dir + '/build/bin;'
    import pyopenpose as op

    params = {'model_folder': op_dir + '/models',
              'face': True, 'face_detector': 2, 'face_detector':0}

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    videos_dir = 'videos'
    video_name = '20190214_Group1-1_cropped.mp4'
    cap = cv2.VideoCapture(videos_dir+'/'+video_name)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    fc = Frame_rate_calculator()

    fc.start_record()
    ret, frame = cap.read()

    # write video
    output_videos_dir = 'result/videos'
    if not os.path.exists(output_videos_dir):
        os.makedirs(output_videos_dir)
        print('*** create directory {} ***'.format(output_videos_dir))
    output_video_name = '.'.join(video_name.split(
        '.')[:-1]) + '_op' + '.' + video_name.split('.')[-1]
    wrt = cv2.VideoWriter(os.path.join(
        output_videos_dir, output_video_name), 0x00000021, fps, (width, height))

    # store keypoint
    face_keypoints = []
    body_keypoints = []
    face_keypoint_file_name = '.'.join(output_video_name.split('.')[:-1])+'_face.pkl'
    body_keypoint_file_name = '.'.join(output_video_name.split('.')[:-1])+'_body.pkl'

    # progress bar
    pbar = tqdm(total=frames)

    while ret:

        datum = op.Datum()
        datum.cvInputData = frame
        time1 = time.time()
        # print(time.time()-time1)

        opWrapper.emplaceAndPop([datum])

        # # detect
        img = np.copy(datum.cvOutputData)

        face_keypoints.append(datum.faceKeypoints)
        body_keypoints.append(datum.poseKeypoints)

        cv2.putText(img, 'FPS:'+str(fc.get_frame_rate()), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        fc.frame_end(time.time())

        wrt.write(img)
        pbar.update()

        cv2.imshow('i', img)

        if cv2.waitKey(1) == 27:
            exit(0)

        ret, frame = cap.read()
    pbar.close()
    wrt.release()
    with open(os.path.join(output_videos_dir, face_keypoint_file_name), 'wb') as f:
        print('*** storing face_keypoints({}) ***'.format(len(face_keypoints)))
        pickle.dump(face_keypoints, f)
    
    with open(os.path.join(output_videos_dir, body_keypoint_file_name), 'wb') as f:
        print('*** storing body_keypoints({}) ***'.format(len(body_keypoints)))
        pickle.dump(body_keypoints, f)
    


if __name__ == "__main__":
    main()
