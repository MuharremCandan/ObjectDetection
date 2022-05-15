import urllib
# import tensorflow as tf
# import tensorflow_hub as hub
import urllib.request

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image



def object_detection_image():
    st.title('Resim İçin Nesne Tanıması')
    st.subheader("""
    Verilen resimdeki fareyi tanıyan ve onu bounding box ile çerçeveleyen yolo algoritması projesi.
    """)
    file = st.file_uploader('Resim Yükle', type=['jpg', 'png', 'jpeg'])
    if file != None:
        img1 = Image.open(file)
        img2 = np.array(img1)

        st.image(img1, caption="Yüklenen Resim")
        my_bar = st.progress(0)
        confThreshold = st.slider('Kesinlik %', 0, 100, 50)
        nmsThreshold = st.slider('Eşik Değeri', 0, 100, 20)

        whT = 320
        classNames = ["mouse"]
        
        config_path = r'config_n_weights\yolov4_ders.cfg'
        weights_path = r'config_n_weights\yolov4_ders_last.weights'
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        def findObjects(outputs, img):
            hT, wT, cT = img2.shape
            bbox = []
            classIds = []
            confs = []
            for output in outputs:
                for det in output:
                    scores = det[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > (confThreshold / 100):
                        w, h = int(det[2] * wT), int(det[3] * hT)
                        x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                        bbox.append([x, y, w, h])
                        classIds.append(classId)
                        confs.append(float(confidence))

            indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold / 100, nmsThreshold / 100)
            obj_list = []
            confi_list = []
            # drawing rectangle around object
            for i in indices:
                i = i
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                # print(x,y,w,h)
                cv2.rectangle(img2, (x, y), (x + w, y + h), (240, 54, 230), 2)
                # print(i,confs[i],classIds[i])
                obj_list.append(classNames[classIds[i]].upper())

                confi_list.append(int(confs[i] * 100))
                cv2.putText(img2, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 0, 240), 2)
            df = pd.DataFrame(list(zip(obj_list, confi_list)), columns=['Nesne İsmi', 'Kesinlik %'])
            st.write(df)
            
                

        blob = cv2.dnn.blobFromImage(img2, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs, img2)

        st.image(img2, caption='Saptanmış Resim.')

        cv2.waitKey(0)

        cv2.destroyAllWindows()
        my_bar.progress(100)
