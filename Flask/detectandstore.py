from flask import Flask,render_template,Response,request
import cv2
import tensorflow as tf
import numpy as np

face_cascade = cv2.CascadeClassifier("Models/haarcascade_frontalface_default.xml")
model = tf.keras.models.load_model("Models/80_79.h5")
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detectandupdate(img):
    path = "static/" + str(img)
    image = cv2.imread(path)
    # image=tf.io.read_file(path)
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in faces:
        x += 20
        w -= 40
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi = gray_frame[y:y+h, x:x+w]
        # roi = tf.image.resize(roi, size = [100, 100])
        roi = cv2.resize(roi, (100,100))
        # roi = tf.convert_to_tensor(roi, dtype=tf.32)
        # roi = tf.image.resize(roi, size = [100, 100])
        # roi=tf.io.decode_image(tf.Variable(roi),channels=3)
        roi=tf.expand_dims(roi,axis=-1).numpy()
        roi=cv2.cvtColor(roi,cv2.COLOR_GRAY2RGB)
        pred = model.predict(tf.expand_dims(roi, axis=0))
        pred_class = class_names[tf.argmax(pred, axis = 1).numpy()[0]]
        cv2.putText(image, pred_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        path = "static/" + "pred" + str(img)
        cv2.imwrite(path, image)


    return ([img, "pred" + img])