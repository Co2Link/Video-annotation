import tkinter as tk
import tkinter.filedialog
import cv2
from tkinter.ttk import Progressbar
from PIL import ImageTk,Image
import os
import configparser

import keyboard

from utils import image_resize

# Setting
config = configparser.ConfigParser()
config.read('config.ini')
Person = dict(config['PERSON'])
Action = dict(config['ACTION'])
Setting = dict(config['SETTING'])

DELAY = int(Setting['delay'])

WINDOW_SIZE = (int(Setting['window_width']),int(Setting['window_height']))
CANVAS_SIZE = (int(Setting['canvas_width']),int(Setting['canvas_height']))


class App:
    def __init__(self, window):
        self.window = window
        self.window.title('Annotation')

        # Menu
        menubar = tk.Menu(window)
        self.window.config(menu=menubar)
        # file open menu
        fileMenu = tk.Menu(menubar)
        fileMenu.add_command(label="Open",command=self.onOpen)
        menubar.add_cascade(label="FIle",menu=fileMenu)

        # Create a canvas that can fit the video
        self.canvas = tk.Canvas(window, width = CANVAS_SIZE[0], height = CANVAS_SIZE[1],bg='black')
        self.canvas.pack()

        # Progress bar
        self.progress_bar = Progressbar(window,orient = tk.HORIZONTAL,length=1280,maximum=100,mode='determinate')
        self.progress_bar.pack()

        # Status label show that if it is labeling
        self.status_label = tk.Label(window,text='noob',bg='white',font=("Courier", 44))
        self.status_label.pack(side=tk.LEFT)

        # Play button
        self.play = False 
        self.play_btn_text = tk.StringVar(window,value='Play')
        self.play_btn = tk.Button(window,textvariable=self.play_btn_text,command=self.play_or_stop_video)
        self.play_btn.pack(padx=5,pady=5)
       
        # Fastforward/Fastbackward button
        btn_frame_1 = tk.Frame(window)
        btn_frame_1.pack(fill='y',expand=False)
        # use lambda to pass arugument
        tk.Button(btn_frame_1,text='fast backward',command=lambda : self.onClick_fastForwardBackward(backward=True)).grid(row=0,column=1,padx=5,pady=5)
        tk.Button(btn_frame_1,text='fast forward',command=self.onClick_fastForwardBackward).grid(row=0,column=2,padx=5,pady=5)

        # Labels that show information of video
        lb_frame = tk.Frame(window)
        lb_frame.pack(fill='y',expand=False)
        self.fps_text = tk.StringVar(window,value='FPS')
        self.duration_text = tk.StringVar(window,value='Time')
        tk.Label(lb_frame,textvariable=self.fps_text).grid(row=0,column=1,padx=5,pady=5)
        tk.Label(lb_frame,textvariable=self.duration_text).grid(row=0,column=2,padx=5,pady=5)

        # Radiation button that set the label's information
        conf_frame = tk.Frame(window)
        conf_frame.pack(fill='y',expand=False)
        conf_sub_frame_left = tk.Frame(conf_frame)
        conf_sub_frame_left.grid(row=0,column=1,padx=10,pady=5)
        tk.Label(conf_sub_frame_left,text='Select Person: ').pack(side=tk.LEFT)
        self.person_variable = tk.IntVar()
        for key,value in Person.items():
            tk.Radiobutton(conf_sub_frame_left,text=key,padx=5,variable=self.person_variable,command=self._showchoice,value=int(value)).pack(side=tk.LEFT)
        conf_sub_frame_right = tk.Frame(conf_frame)
        conf_sub_frame_right.grid(row=0,column=2,padx=10,pady=5)
        conf_sub_frame_right.grid(row=0,column=2,padx=10,pady=5)
        tk.Label(conf_sub_frame_right,text='Select Action: ').pack(side=tk.LEFT)
        self.action_variable = tk.IntVar()
        for key,value in Action.items():
            tk.Radiobutton(conf_sub_frame_right,text=key,padx=5,variable=self.action_variable,command=self._showchoice,value=int(value)).pack(side=tk.LEFT)

        # Begin to 
        self.window.mainloop()

    def _showchoice(self):
        """ DEBUG """
        print('*** Selected label ***')
        print('Person: {}, Action: {}'.format(list(Person.keys())[self.person_variable.get()],list(Action.keys())[self.action_variable.get()]))

    def play_or_stop_video(self):
        """ Play or pause video """
        if hasattr(self,'vid'):
            if self.play:
                self.play=False
                self.play_btn_text.set('PLAY')
                self.window.after_cancel(self.after_id)
            else:
                self.play=True
                self.play_btn_text.set('PAUSE')
                self.play_video()


    
    def onOpen(self):
        """ Open video file """
        self.video_path = tk.filedialog.askopenfilename(initialdir=os.path.dirname(os.path.abspath(__file__)),
                                filetypes =(("Video File", "*.mp4"),("All Files","*.*")),
                                title = "Choose a file."
                                )
        print ('*** open file ***')
        print(self.video_path)
        try:
            self.vid = cv2.VideoCapture(self.video_path)
            # increase buffer for faster reading speed
            self.vid.set(cv2.CAP_PROP_BUFFERSIZE,100)
            
            self.vid_info={
                'fps':self.vid.get(cv2.CAP_PROP_FPS),
                'frames':self.vid.get(cv2.CAP_PROP_FRAME_COUNT),
                'duration':round(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)/self.vid.get(cv2.CAP_PROP_FPS),1)
            }
            print('*** video information ***')
            print(self.vid_info)
            self._showchoice()
            self.frame_labels = [False for _ in range(int(self.vid_info['frames']))]
            self.pos_frame = 0
            self.progress_bar['maximum'] = int(self.vid_info['frames'])
            self.progress_bar['value'] = 1
            self.play_video(one_frame=True)
            
        except Exception as e:
            print(e)
            print("No file exists")
    
    def onClick_fastForwardBackward(self,step = 100,backward = False):
        """ Fastforward or fastbackward by step frames """
        if hasattr(self,'vid'):
            if backward:
                next_pos_frame = self.pos_frame - step
                for idx in range(next_pos_frame,self.pos_frame+1):
                    self.frame_labels[idx-1] = False
            else:
                next_pos_frame = self.pos_frame + step 


            if next_pos_frame < 1:
                self.pos_frame = 1
            elif next_pos_frame >= int(self.vid_info['frames']):
                self.pos_frame = int(self.vid_info['frames'])
            else:
                self.pos_frame = next_pos_frame

            # the next frame from read() locate at n+1
            self.pos_frame -= 1
            self.vid.set(cv2.CAP_PROP_POS_FRAMES,self.pos_frame)
            self.play_video(one_frame=True)

    def _update_info_label(self):
        """ Update the information using on gui """
        progress_percentage = self.pos_frame/self.progress_bar['maximum']
        progress_second = int(self.vid_info['duration']*progress_percentage)
        minutes = int(progress_second/60)
        seconds = progress_second%60
        self.fps_text.set('Frame: {}/{}'.format(self.pos_frame,self.progress_bar['maximum']))
        self.duration_text.set('Time: {}:{}/{}:{}'.format(minutes,seconds,int(self.vid_info['duration']/60),int(self.vid_info['duration']%60)))

    def _save_labels(self):
        """ Save the labels as csv file at the end of the video """
        if not os.path.exists('result'):
            os.mkdir('result')
            print('*** create directory /result ***')

        file_name='{}_{}_{}.csv'.format(os.path.basename(self.video_path),list(Person.keys())[self.person_variable.get()],list(Action.keys())[self.action_variable.get()])

        with open('result/'+file_name,'w') as f:
            print('*** saving result at result/{} ***'.format(file_name))
            f.writelines([str(1 if label else 0)+'\n' for label in self.frame_labels])
        
    def play_video(self,one_frame = False):  
        """ Play video with frame delay of DELAY ms"""
        ret,frame = self.vid.read()

        if ret:
            # resize image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # resize img without distortion, time cost is much lower than PIL.Image.thumbnail
            img = image_resize(frame,width=CANVAS_SIZE[0],height=CANVAS_SIZE[1])
            img = Image.fromarray(img)


            self.photo = ImageTk.PhotoImage(image = img)
            self.canvas.create_image(0,0,image = self.photo,anchor = tk.NW)

            # Progress bar
            self.pos_frame += 1
            self.progress_bar['value'] = self.pos_frame
            self._update_info_label()


            if keyboard.is_pressed('space'):
                self.frame_labels[self.pos_frame-1] = True
                self.status_label.config(text='label',bg='red')
            else:
                self.status_label.config(text='noob ',bg='white')

            if not one_frame:
                if self.play:
                    self.after_id = self.window.after(DELAY,self.play_video)
                else:
                    self.window.after_cancel(self.after_id)
                

        else:
            self.play_or_stop_video()
            self._save_labels()

window = tk.Tk()
window.geometry("{}x{}".format(*WINDOW_SIZE))

App(window)