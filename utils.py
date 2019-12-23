import cv2
import numpy as np
import time
from tqdm import tqdm
from collections import deque
from functools import wraps

def timethis(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        print("{} {:.3f}".format(func.__name__,end-start))
        return result
    return wrapper

def resize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    buf = video_path.split('.')
    resized_video_path = '.'.join(buf[:-1])+'_resized'+'.'+buf[-1]
    print(resized_video_path)
    out = cv2.VideoWriter(resized_video_path, fourcc,59.94,(1920,630))

    print('*** resizing video ***')
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret,frame=cap.read()
        if ret == True:
            out.write(frame[:630,:,:])
        else:
            break

    print(cap.get(cv2.CAP_PROP_POS_FRAMES))

    cap.release()
    out.release()
    cv2.destroyAllWindows()



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


class Frame_rate_calculator:
    def __init__(self):
        self.start = 0
        self.time_list = deque(maxlen=10)
        self.current_frame_rate = 0
        self.output_frame_rate = 0
        self.counter = 0
        self.frame_rate_update_time = 0
    def start_record(self):
        self.start = time.time()
    def frame_end(self,now):
        self.time_list.append(now-self.start)
        self.start = now
        self.current_frame_rate = round(1/(sum(self.time_list)/len(self.time_list)),2)
        return self.current_frame_rate
    def get_frame_time(self):
        return sum(self.time_list)/len(self.time_list)
    def get_frame_rate(self):
        if time.time() - self.frame_rate_update_time > 1:
            self.output_frame_rate = self.current_frame_rate
            self.frame_rate_update_time = time.time()
        return self.output_frame_rate

if __name__ == "__main__":
    video_2='20190214_Group1-3.MP4'
    resize_video('C:/Users/82503/Desktop/'+video_2)