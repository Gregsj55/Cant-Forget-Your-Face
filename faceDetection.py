import cv2
import pickle

cascPath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.read("trainer.yml")
labels = {}
with open("labels.pickle", 'rb') as f:
    labels_old = pickle.load(f)
    labels = {v:k for k,v in labels_old.items()}

#cap = cv2.VideoCapture(0)

#cap.set(3,640)
#cap.set(4,480)
counter = 0
while True:
    #print(counter)
    img = cv2.imread("test-images/oprah.jpeg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w] #grayscale face

        id_, conf = recognizer.predict(face)
        print(conf)
        if conf <= 70: # if model thinks it is certain person, put name on box
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Picture", img)
    counter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cap.release()
cv2.destroyAllWindows()