from analysis import read_input
import pandas as pd
import cv2
from collections import deque
from analysis import get_model

def input_preprosess(inputs):
    """ format data from dict"""
    columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r','AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
    X = pd.DataFrame()

    for video in ['1','2','3','4']:
        for person in ['a','b','c']:
            df = inputs[video][person]
            inputs[video][person] = df[columns]

    return inputs

def rule(inputs):
    """
    args:
        inputs
    
    output:
        list
    """
    


def main():
    buf = deque(maxlen=10)

    inputs = input_preprosess(read_input())
    model = get_model()
    result = model.predict(inputs['4']['a'])

    cap = cv2.VideoCapture('videos/20190218_Group3-2_cut_cropped_a.mp4')

    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == len(inputs['4']['a'])

    ret,frame = cap.read()


    while ret:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
        buf.append(result[idx])
        if sum(buf)/len(buf) >= 0.5:
            text = 'smile'
        else:
            text = 'no smile'

        frame = cv2.putText(frame,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        cv2.imshow('putText',frame)

        ret,frame = cap.read()



        if cv2.waitKey(10) == 27:
            exit(0)


if __name__ == "__main__":
    main()