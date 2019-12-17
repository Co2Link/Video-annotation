import wave
import pygame
import time
file_path='audio/file_example_WAV_1MG.wav'




import pygame
from tkinter import *
root = Tk()
# pygame.init()
def play():
    file_wav = wave.open(file_path)
    frequency = file_wav.getframerate()
    pygame.mixer.init(frequency=5000)
    print(pygame.mixer.get_init())
    pygame.mixer.music.load('audio/file_example_WAV_1MG.wav') #Loading File Into Mixer
    pygame.mixer.music.play() #Playing It In The Whole Device
    time.sleep(10)
    pygame.mixer.music.pause()
Button(root,text="Play",command=play).pack()
root.mainloop()