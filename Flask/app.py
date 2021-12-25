from flask import Flask,render_template,Response,request
import cv2
import tensorflow as tf
import numpy as np
import os
import shutil
from detectandstore import detectandupdate
from pushfile import pushIntoFile
from werkzeug.utils import secure_filename
from flask import current_app
from flask import send_file


face_cascade = cv2.CascadeClassifier("Models/haarcascade_frontalface_default.xml")
model = tf.keras.models.load_model("Models/80_79.h5")
app=Flask(__name__)
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}



def MakeZipFile(filepath):
    shutil.make_archive('./dataset','zip',filepath)

def MakeZipLabel(filepath):
    shutil.make_archive('./labeldata','zip',filepath)


def generate_frames():
    camera=cv2.VideoCapture(0)

    while True:
            
        success,frame=camera.read()
        img = frame.copy()
        
        if not success:
            break
        else:
            original = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                x += 20
                w -= 40
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                image = original[y:y+h, x:x+w]
                image = tf.image.resize(image, size = [100, 100])
                pred = model.predict(tf.expand_dims(image, axis=0))
                pred_class = class_names[tf.argmax(pred, axis = 1).numpy()[0]]
                cv2.putText(frame, pred_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/webcam')
def webcam():
    return render_template('index.html')

def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

@app.route('/detectpic', methods=['GET', 'POST'])
def detectpic():
    UPLOAD_FOLDER = 'static'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':

        file = request.files['file']

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            result =detectandupdate(filename)

            return render_template('showdetect.html', orig=result[0], pred=result[1])

@app.route('/picdetect')
def picdetect():
    return render_template('picdetect.html')

@app.route('/bulkdetect', methods=['GET', 'POST'])
def bulkdetect():
    UPLOAD_FOLDER = './preparedataset/input'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':
        for file in request.files.getlist('file'):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                pushIntoFile(filename)
        MakeZipFile('./preparedataset')
        return render_template('donedow.html')

@app.route('/red_to_bulkin')
def red_to_bulkin():
    return render_template('bulkinput.html')


@app.route('/makebound', methods=['GET', 'POST'])
def makebound():
    UPLOAD_FOLDER = './labeleddata/input'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if(request.method == 'POST'):
        for file in request.files.getlist('file'):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                path = "labeleddata/input/" + str(filename)
                image = cv2.imread(path)
                gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces= face_cascade.detectMultiScale(gray_frame, 1.3, 5)
                for (x, y, w, h) in faces:
                    x += 20
                    w -= 40
                    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                    filepath="labeleddata/output/"+ str(filename)[:-4]+'.txt'
                    with open(filepath, 'w') as f:
                        content=str(x) + ' ' + str(y) + ' ' + str(w) + ' '+ str(h)
                        f.write(content)
        MakeZipLabel('./labeleddata')          
    return render_template('donelabel.html')

@app.route('/getinlab')
def getinlab():
    return render_template('inputforbound.html')


@app.route('/downloadDS', methods=['GET', 'POST'])
def downloaddataset():
    path='dataset.zip'
    return send_file(path,as_attachment=True)

@app.route('/downloadLB', methods=['GET', 'POST'])
def downloadLabel():
    path='labeldata.zip'
    return send_file(path,as_attachment=True)


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)