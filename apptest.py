#import sqlalchemy
from imutils.video import VideoStream
from flask import Response, Flask, render_template, request, send_from_directory, redirect, url_for
from flask_migrate import Migrate
import threading
import argparse
import datetime
import imutils
import time
import cv2
import os
import numpy as np
from PIL import Image
import pickle
from flask_sqlalchemy import SQLAlchemy
import psycopg2
import io
import base64
from flask_socketio import SocketIO, emit
import sys
from flask_cors import CORS, cross_origin
from camera import Camera
import boto3
import aws_config
import os.path




app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://lnisnsukhcqzop:bb96976d926974387df7c49d4ff570a73b00e1b02629cba11687444dfa15ee14@ec2-34-230-198-12.compute-1.amazonaws.com:5432/d7nct4sujlsnie'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:pwd@db:5432/my_db'

db = SQLAlchemy(app)

CORS(app)
socketio = SocketIO(app, always_connect=True, engineio_logger=False)
camera = Camera()

uName = ""
tempUName = ""
faceDetected = False
recognizer = None
labels = {}
labels_old = {}
snapflag = False
trainflag = False
trainedFlag = False
uFolder = ""
foldCount = 0

MIGRATE = Migrate(app, db)
PYTHONUNBUFFERED=True

aws_config.setVar()

# s3 = boto3.client('s3')
# s3.put_object(Bucket="cant-forget-your-face", Key='tmp/')
#
# if not os.path.isdir('tmp/'):
#     os.mkdir('tmp/')

# with app.app_context():
#     db.create_all()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)

    password = db.Column(db.String(80), unique=False, nullable=False)
    text = db.Column(db.Text(), unique=False, nullable = True)
    #model = db.Column(db.Blob(), unique=False, nullable = True)

    def __init__(self, username=None, password=None, text = None):
        self.username = username
        self.password = password
        self.text = text



def detect_face():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "Resources")  # grab image directory path

    cascPath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    recognizer = cv2.face_LBPHFaceRecognizer.create()

    current_id = 0
    label_ids = {}

    x_train = []
    y_labels = []

    for root, dir, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("PNG") or file.endswith("JPG"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()  # set label to folder name
                print(label, path)
                if not label in label_ids:
                    label_ids[label] = current_id  # add label to dictionary
                    current_id += 1  # value of dict
                id_ = label_ids[label]
                # print(label_ids)
                # y_label.append(label)
                # x.append.append(path)
                pil_image = Image.open(path).convert("L")
                size = (550, 550)
                res_image = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(res_image, "uint8")  # convert image to numpy array for training
                # print(image_array)
                faces = faceCascade.detectMultiScale(image_array, 1.1, 4)
                for (x, y, w, h) in faces:
                    face = image_array[y:y + h, x:x + w]
                    x_train.append(face)  # append face to model
                    y_labels.append(id_)  # append label

    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainer.yml")




    vs = VideoStream(src=0).start()
    cascPath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read("trainer.yml")
    labels = {}
    with open("labels.pickle", 'rb') as f:
        labels_old = pickle.load(f)
        labels = {v: k for k, v in labels_old.items()}

    while True:
        frame = vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            id_, conf = recognizer.predict(face)
            print(conf)
            if conf >= 65 and conf <= 85:  # if model thinks it is certain person, put name on box
                print(id_)
                print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.imshow("vid",img)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@socketio.on('image')
def image(data_image, username, capflag, snapflag, verifyflag):
    global faceDetected
    global foldCount, uFolder, trainflag, trainedFlag, tempUName
    tempUName = username

    input = data_image.split(",")[1]
    camera.enqueue_input(input, username, capflag, snapflag, verifyflag)

    #print(username, file=sys.stderr)

    # headers, b64_image = data_image.split(',', 1)
    # sbuf = io.StringIO()
    # sbuf.write(b64_image)
    #
    # b = io.BytesIO(base64.b64decode(b64_image))
    #
    # pimg = Image.open(b)
    #
    # frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    #
    # if (uFolder != "" and foldCount <= 10 and trainflag):
    #     pic = imutils.resize(frame, width=700)
    #     print('capturing images', file=sys.stderr)
    #     path = './' + uFolder
    #     cv2.imwrite(os.path.join(path , username + str(foldCount) + '.png'), pic)
    #     foldCount += 1
    # if foldCount > 10:
    #     trainflag = False
    #     trainedFlag = True
    #     foldCount = 0
    #
    # if snapflag == True:
    #     snapflag = False
    #     # print("taking pic to be verified", flush=True)
    #     # print(username, flush=True)
    #     # print('./.' + username + '/verifypic.png', flush=True)
    #     cv2.imwrite('./.' + username + '/verifypic.png', imutils.resize(frame, width=700))
    #
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    #
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    #
    #
    #
    # imgencode = cv2.imencode('.jpg', frame)[1]
    # stringdata = base64.b64encode(imgencode).decode('utf-8')
    # b64_src = 'data:image/jpg;base64,'
    # stringdata = b64_src + stringdata
    #
    # emit('response_back', stringdata)


#Home Page landing
@app.route('/')
def index():
    #return 'Home Page'
    # print('Hello Home!', file=sys.stderr)
    return render_template("index.html")


@app.route('/snap')
def snap():
    print('Snap function called', file=sys.stderr)
    global snapflag
    snapflag = True
    return "nothing"




@app.route('/verify')
def verify():
    global faceDetected,tempUName
    print(tempUName, flush=True)
    print('veryify function called', file=sys.stderr)
    global recognizer
    global labels
    global labels_old
    cascPath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read("./." + tempUName + "/trainer.yml")
    with open("./." + tempUName + "/labels.pickle", 'rb') as f:
        labels_old = pickle.load(f)
        print(labels_old, file=sys.stderr)
        labels = {v: k for k, v in labels_old.items()}
    img = cv2.imread("./." + tempUName + '/verifypic.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]  # grayscale face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('./.' + tempUName + '/faces.png', img)
        id_, conf = recognizer.predict(face)
        if(conf < 95):
            faceDetected = True
        print("conf", flush=True)
        print(conf, flush=True)
    return "nothing"



@app.route('/<path:path>')
def serve(path):
    print('Path:', file=sys.stderr)
    print(path, file=sys.stderr)
    if path == 'login.html':
        print('Serving Login Page', file=sys.stderr)
    # global uName
    # if path == 'createAcc.html' and uName != "":
    #     user = User.query.filter_by(username=uName).first()
    #     return render_template("TextEdit.html", text = user.text)

    return render_template(path)
    

def train(username):
    s3 = boto3.client('s3')
    image_dir = 'tmp/' + username  # grab image directory path

    cascPath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    recognizer = cv2.face_LBPHFaceRecognizer.create()

    current_id = 0
    label_ids = {}

    x_train = []
    y_labels = []

    for root, dir, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("PNG") or file.endswith("JPG"):
                path = os.path.join(root, file)
                s3.upload_file('tmp/' + username + '/' + file, "cant-forget-your-face", 'tmp/' + username + '/' + file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()  # set label to folder name
                print(label, path)
                if not label in label_ids:
                    label_ids[label] = current_id  # add label to dictionary
                    current_id += 1  # value of dict
                id_ = label_ids[label]
                # print(label_ids)
                # y_label.append(label)
                # x.append.append(path)
                pil_image = Image.open(path).convert("L")
                size = (550, 550)
                res_image = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(res_image, "uint8")  # convert image to numpy array for training
                # print(image_array)
                faces = faceCascade.detectMultiScale(image_array, 1.1, 4)
                for (x, y, w, h) in faces:
                    face = image_array[y:y + h, x:x + w]
                    x_train.append(face)  # append face to model
                    y_labels.append(id_)  # append label

    with open(image_dir + "/labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save(image_dir + "/trainer.yml")
    s3.upload_file('tmp/' + username + '/trainer.yml', "cant-forget-your-face", 'tmp/' + username + '/trainer.yml')


def download_file(file_name, bucket):
    """
    Function to upload a file to an S3 bucket
    """

    object_name = file_name
    s3_client = boto3.client('s3')
    s3_client.download_file("cant-forget-your-face", 'tmp/joe1/trainer.yml', 'tmp/joe1/trainer.yml')
    print("download", flush=True)
    #response = s3_client.upload_file(file_name, bucket, object_name)
    #s3_client.put_object(Bucket="cant-forget-your-face", Key="matt/")
    return "nothing"


@app.route('/test')
def test():
    print("test", flush=True)

    download_file("0.png", 'cant-forget-your-face')
    #data = open('0.png', 'rb')
    #s3.Bucket('cant-forget-your-face').put_object(Key='0.png', Body=data)
    return "nothing"



def createAndSave():
    global tempUName, foldCount, uFolder
    os.mkdir("." + tempUName)
    uFolder = "." + tempUName
    foldCount = 0
#     text = User.query.filter_by(username = user).

@app.route("/saveImages")
def saveImages():
    createAndSave()
    print('Save Images function called', file=sys.stderr)
    global trainflag
    trainflag = True
    return "nothing"




#Register Page Action: Register
@app.route('/createAcc.html', methods = ['POST'])
def register_post():
    global trainflag, trainedFlag, uName
    db.create_all()
    db.session.commit()
    # if uName != "":
    #     user = User.query.filter_by(username=uName).first()
    #     return render_template("TextEdit.html", text = user.text)
    # User.query.all()
    usern = request.form['username']
    pass1 = request.form['password1']
    pass2 = request.form['password2']
    if usern == "" or pass1 == "":
        return "must not be empty"
    user = User.query.filter_by(username=usern).first()
    if user is not None:
        return "account name already in use"
    # user = User.query.filter_by(password = pass1).first()
    # if user is not None:
    #     return "account password already in use"
    if pass1 == pass2:
        newUser = User(username = usern, password = pass1, text = "Type in this box")
        db.session.add(newUser)
        db.session.commit()
    else:
        return "pass not same"
    user = User.query.filter_by(username=usern).first()
    if user is None:
        return "err: user not created"
    # if trainedFlag is False:
    #     return "err: picture not captured"
    else:
        train(user.username)
        uName = user.username
        # createAndSave()
        trainflag = True
        return render_template("TextEdit.html", text = user.text) #redirect(url_for("http://localhost:5000/textEdit.html", text = "user text"))


def gen():

    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/login", methods = ['POST'])
def login():
    global faceDetected
    global uName


    db.create_all()
    db.session.commit()

    form_username = request.form['username']
    form_password = request.form['password']
    
    user = User.query.filter_by(username = form_username).first()

    if(user is None):
        return "Username doesn't exist"
    # if not faceDetected:
    #     return("Login failed, face not detected")
    if(user.password == form_password):
        uName = user.username
        
        return render_template("TextEdit.html", text = user.text)
    else:
        return "Incorrect password"

@app.route("/save", methods = ['POST'])
def save():
    global uName
    # db.create_all()
    # db.session.commit()
    
    user = User.query.filter_by(username = uName).first()

    if user is None:
        return "it broke"

    user.text = request.form['classic']
    db.session.commit()
    return render_template("TextEdit.html", text = User.query.filter_by(username = uName).first().text)

@app.route("/logout", methods = ['POST'])
def logout():
    global uName
    global faceDetected
    faceDetected = False
    uName = ""
    return redirect("/index.html")

@app.route("/loginFirst", methods = ['POST'])
def loginFirst():
    global uName
    if uName != "":
        user = User.query.filter_by(username = uName).first()
        return render_template("TextEdit.html", text = user.text)
    else:
        return redirect("/login.html")

# @app.route("/saveImgs", methods = ['POST'])
# def saveImgs():
    

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(debug=True, host='0.0.0.0', port=port)