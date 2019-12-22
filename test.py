import wave
import pygame
import time
file_path='audio/20190214_Group1-1_cropped.wav'




import pygame
from tkinter import *
root = Tk()
# pygame.init()
def play():
    file_wav = wave.open(file_path)
    frequency = file_wav.getframerate()
    print(frequency)
    pygame.mixer.init(frequency=5000)
    print(pygame.mixer.get_init())
    pygame.mixer.music.load(file_path) #Loading File Into Mixer
    pygame.mixer.music.play() #Playing It In The Whole Device
    time.sleep(10)
    pygame.mixer.music.pause()
Button(root,text="Play",command=play).pack()
root.mainloop()