<h1 align="center">Face Mask Detection</h1>

- - -

<div align= "center">
  <h4>Face Mask Detection system built with OpenCV, Keras/TensorFlow using Deep Learning and Computer Vision concepts in order to detect face masks in static images and static
    videos. Also it is as well as in real-time video streams. </h4>
</div>

- - -

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

## 👉 Which Tech & framework used ?

- [OpenCV](https://opencv.org/)
- [OpenH264](https://https://github.com/cisco/openh264)
- [Face detector Model](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [Streamlit](https://www.streamlit.io/)

<br/>

## 🔥 What is Streamlit?
Streamlit is an open-source Python library that makes it easy to build beautiful custom web-apps for machine learning and data science. To use it, just pip install streamlit , then import it, write a couple lines of code, and run your script with streamlit run [filename] <br/>
<p align="center"><img src="https://user-images.githubusercontent.com/47052106/90765856-a46b1e00-e325-11ea-9b2a-549fb4f96151.png" width="400" height="200"></p>

<br/>

## 🌈 Introduction
In the present scenario due to Covid-19, there is no efficient face mask detection applications which are now in high demand for transportation means, densely populated areas, residential districts, large-scale manufacturers and other enterprises to ensure safety. Also, the absence of large datasets of __‘with_mask’__ images has made this task more cumbersome and challenging. 

<br/>

## ⚡️ Project Demo
- Static Image
<br/>

![image](https://user-images.githubusercontent.com/47052106/90777920-fa948d00-e336-11ea-9f59-15861f5c84ee.JPG)

<br/>

- Static Video
<br/>

![video](https://user-images.githubusercontent.com/47052106/90778045-20219680-e337-11ea-9b01-77c9a7864fda.JPG)

<br/>

- Realtime - Webcam
<br/>

<br/>

## 📁 Dataset
The dataset used can be downloaded here - [Click to Download](https://drive.google.com/drive/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG?usp=sharing)

This dataset consists of __3835 images__ belonging to two classes:
*	__with_mask: 1916 images__
*	__without_mask: 1919 images__

The images used were real images of faces wearing masks. The images were collected from the following sources:

* __Bing Search API__ ([See Python script](https://github.com/Hott-J/Face-Mask-Detection/blob/master/search.py))
* __Kaggle datasets__ 
* __RMFD dataset__ ([See here](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset))

## 📌 Prerequisites

All the dependencies and required libraries are included in the file <code>requirements.txt</code> [See here](https://github.com/Hott-J/Face-Mask-Detection/blob/master/requirements.txt)

<br/>

## 🚀 How to Install
1. Clone the repo
```
$ git clone https://github.com/Hott-J/Face-Mask-Detection.git
```

2. Change your directory to the cloned repo and create a Python virtual environment named 'test'
```
$ mkvirtualenv test
```

3. Install the libraries required
```
$ pip3 install -r requirements.txt / pip install -r requirements.txt
```

## 💥 How to Run

1. Go into the cloned project directory folder and type the following command:
```
$ python3 train_mask_detector.py --dataset dataset
```

2. To detect face masks in a static image, type the following command: 
```
$ python3 detect_mask_image.py --image images/pic1.jpeg
```

3. To detect face masks in a static video streams, type the following command:
```
$ python3 detect_mask_video.py 
```

<br/>

## 🍭 Results

#### This Model gave 93% accuracy for Face Mask Detection after training via <code>tensorflow-gpu==2.0.0</code>

![](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/Readme_images/Screenshot%202020-06-01%20at%209.48.27%20PM.png)

#### We got the following accuracy/loss training curve plot
![](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/plot.png)

<br/>

## 🐶 How to Run in Streamlit Webapp

1. Go into the cloned project directory folder and type the following command:
```
$ streamlit run app.py 
```

<br/>

## ☘️ Finish!
Feel free to mail me for any query! Thank you ❤️
