import threading, cv2, io, base64, os
import os.path
import boto3
from PIL import Image
from time import sleep
import numpy as np
import imutils

class Camera(object):

    def __init__(self):

        self.to_process = []
        self.to_send_back = []
        self.faceCascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face_LBPHFaceRecognizer.create()
        self.trained_images = 0
        self.snappedflag = False
        self.verifiedflag = False
        self.accessflag = False

        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()


    def process_one(self):
        if not self.to_process:
            return

        data = self.to_process.pop(0)
        input_str, username, capflag, snapflag, verifyflag = data
        imgdata = base64.b64decode(input_str)
        input_img = np.array(Image.open(io.BytesIO(imgdata)))
        frame = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        if verifyflag and self.verifiedflag == False:
            self.verifiedflag = True
            self.accessflag = self.verify(frame, username)

        elif snapflag and self.snappedflag == False: #takes pic to be checked
            self.snappedflag = True
            path = 'tmp/' + username
            if not os.path.isdir(path):
                os.mkdir(path)

            s3_client = boto3.client('s3')
            download_path = 'tmp/' + username + '/trainer.yml'
            s3_client.download_file("cant-forget-your-face", download_path, download_path)
            ver_path = 'tmp/' + username + '/verifypic.png'
            pic = imutils.resize(frame, width=900)
            cv2.imwrite(ver_path, pic)


        elif capflag and self.trained_images < 10: #captures images to be trained for user
            if self.trained_images == 0: #create s3 dir for user and add user dir to heroku tmp dir
                s3 = boto3.client('s3')
                s3.put_object(Bucket="cant-forget-your-face", Key=('tmp/' + username + '/'))
                os.mkdir('tmp/' + username)

            path = 'tmp/' + username + '/' + str(self.trained_images) + '.png'
            print(path, flush=True)
            pic = imutils.resize(frame, width=900)
            cv2.imwrite(path, pic)
            self.trained_images += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        buffer = cv2.imencode('.jpeg', frame)[1]
        output_str = buffer.tobytes()

        self.to_send_back.append(output_str)


    def verify(self, frame, username):
        flag = False


        print("pass", flush=True)
        self.recognizer.read('tmp/' + username + '/trainer.yml')
        pic = cv2.imread('tmp/' + username + '/verifypic.png')
        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            id_, conf = self.recognizer.predict(face)
            if conf < 95:
                print("conf", flush=True)
                print(conf, flush=True)
                return True

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.01)


    def enqueue_input(self, input, username, capflag, snapflag, verifyflag):
        data = (input, username, capflag, snapflag, verifyflag)
        self.to_process.append(data)


    def get_frame(self):
        while not self.to_send_back:
            sleep(0.05)
        return self.to_send_back.pop(0)

    def get_acess(self):
        return self.accessflag
