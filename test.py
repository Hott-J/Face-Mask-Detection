import streamlit as st
import numpy as np
import cv2
import os
import io
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import detect_mask_image
import ffmpeg
import imageio
from moviepy.editor import*
from moviepy.video.fx.resize import resize




vid_file=open("./out_video.mp4","rb").read()
st.video(vid_file,start_time=0)


'''
def image_to_video():
        my_clip = VideoFileClip("./out_video.mp4")
        duration = int(my_clip.duration)
        clips=[]
        for i in range(0, duration):
            my_clip.save_frame("./images/picture" + str(i) + ".jpg", i)
            clips.append(ImageClip("./images/picture"+str(i)+".jpg").set_duration(1))
        video=concatenate_videoclips(clips,method="compose")
        video.write_videofile('./images/result.mp4',fps=120)
'''
