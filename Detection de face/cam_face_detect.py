import urllib.request
import cv2
import numpy as np
import sys

# Obtenir les valeurs fournies par l'utilisateur(la cascade)
cascPath = "haarcascade_frontalface_default.xml"

# Création de la cascade et initialisation a la casade de face
faceCascade = cv2.CascadeClassifier(cascPath)

#paramettre de connexion du telephone
urlString = sys.argv[1]
url ='http://'+urlString+'/shot.jpg?'

while True:
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img = cv2.imdecode(imgNp,-1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),2)
    cv2.imshow('test',img)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        exit(0)
