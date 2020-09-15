import streamlit as st
from PIL import Image , ImageEnhance
import numpy as np
import cv2
import os
import io
import base64
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


st.set_option('deprecation.showfileUploaderEncoding', False)

cam=1

def mask_detection():
    global flag
    flag=0
    # load our serialized face detector model from disk
    global image
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    model = load_model("./models/200914model.h5")

    # load the input image from disk and grab the image spatial
    # dimensions
    image = cv2.imread("./images/out.jpg")
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (hmask, mask, withoutMask) = model.predict(face).squeeze()

			# determine the class label and color we'll use to draw
			# the bounding box and text
            if mask > hmask and mask>withoutMask:
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)
                flag=0
                
            elif hmask>mask and hmask>withoutMask:
                color = (0, 0, 255)
                label = 'Half Mask %d%%' % (hmask * 100)
                flag=1
                
            elif withoutMask>hmask:
                color=(255,0,0)
                label='No Mask %d%%' %(withoutMask * 100)
                flag=2
                
            # include the probability in the label
            #label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            # display the label and bounding box rectangle on the output
			# frame
            #cv2.putText(image, label, (startX, startY - 10),
            #cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            #cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    return flag

def web_cam():
    # Reading webcam (노트북 기본 내장)
    cap = cv2.VideoCapture(0)

    # Reading webcam (외장 - 로지텍 허브 사용)
    #cap = cv2.VideoCapture(0)
    i = 0

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        cv2.imshow('Capturing',img)

        #s를 누르면 해당 경로로 저장
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('./images/filename.jpg',img,params=[cv2.IMWRITE_PNG_COMPRESSION,0])
            cam=0
            break

    cap.release() #카메라 닫기 (필수)
    cv2.destroyAllWindows() #창 전부 닫기 / 추후에 다시 켜기 위함. 해제하지 않으면, 다시 실행할 때 객체가 해제되어있지않아 카메라 다시 안켜짐


def mask_stream():
    st.title("AI Mask Detection")
    st.header("WebCam transfer to Real-Image")
    st.subheader("Dev. HakJae Chung")
    activities = ["Webcam" ,"Image"] #버튼 생성 순서
    choice = st.sidebar.selectbox("Select the type you want to detect",activities) #버튼 생성

    halfmask_sound = open('./sound/halfmask_sound.mp3','rb').read()
    nomask_sound = open('./sound/nomask_sound.mp3','rb').read()

    if choice=="Image": #이미지 버튼 클릭시 동작
        image_file = Image.open("./images/filename.jpg") 
        
        if image_file is not None:
            our_image = image_file
            im=our_image.save("./images/out.jpg")
            saved_image=st.image(image_file,caption='image upload successfully!',use_column_width=True)
    
            if st.button("Process"):
                label = mask_detection()
                if label == 0:
                    st.write("마스크를 잘 쓰고 계시네요.")
                elif label == 1:
                    st.write("마스크를 제대로 착용해 주세요.")
                    mymidia_placeholder = st.empty()
                    mymidia_str = "data:audio/ogg;base64,%s"%(base64.b64encode(halfmask_sound).decode())
                    mymidia_html = """
                                    <audio autoplay class="stAudio">
                                    <source src="%s" type="audio/ogg">
                                    Your browser does not support the audio element.
                                    </audio>
                                    """%mymidia_str

                    mymidia_placeholder.empty()
                    time.sleep(1)
                    mymidia_placeholder.markdown(mymidia_html, unsafe_allow_html=True)

                else:
                    st.write("마스크를 착용해 주세요!")
                    mymidia_placeholder = st.empty()

                    mymidia_str = "data:audio/ogg;base64,%s"%(base64.b64encode(nomask_sound).decode())
                    mymidia_html = """
                                    <audio autoplay class="stAudio">
                                    <source src="%s" type="audio/ogg">
                                    Your browser does not support the audio element.
                                    </audio>
                                    """%mymidia_str

                    mymidia_placeholder.empty()
                    time.sleep(1)
                    mymidia_placeholder.markdown(mymidia_html, unsafe_allow_html=True)

        else:
            st.write("Upload Image!.")

    if choice=="Webcam":
        web_cam()
        st.write("Real-Time-Image file saved!")

mask_stream()