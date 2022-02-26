import numpy as np
import argparse
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import wikipedia


def detect_objects(image1):

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default=image1,
        help="path to input image")
    ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt.txt",
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel",
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())


    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


    print("[INFO] loading model...")
    model = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


    image = cv2.imread(args["image"])
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)


    print("[INFO] computing object detections...")
    model.setInput(blob)
    detections = model.forward()


    for i in np.arange(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]


        if confidence > args["confidence"]:

            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")


            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            resu=0
            try:
                results = wikipedia.summary(CLASSES[idx], sentences = 2)
                print(results)
                resu=(results)
            except:
                print('No matching information formed')
            p = tk.Label(root, text = resu)
            p.pack()



    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    tkimage = ImageTk.PhotoImage(image)
    result.config(image=tkimage)
    result.iamge = tkimage

def search_image():                                                              
    image1 = filedialog.askopenfilename()
    if image1:
        detect_objects(image1)

root = tk.Tk()                                                                   
root.geometry('1500x700')
root.resizable(True,True)
root.title('FunEasyLearn')
w = tk.Label(root, text = "FunEasyLearn", font = "Arial 36", bg ='pink', width = 900)
w.pack()
button = tk.Button(root, text = "CHOOSE", font = "Arial 25",bg='orange', command = search_image)
button.pack()
result = tk.Label(root)
result.pack()
root.mainloop()



