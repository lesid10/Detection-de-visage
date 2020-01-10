import cv2
import sys


# Obtenir les valeurs fournies par l'utilisateur(la cascade)
cascPath = "haarcascade_frontalface_default.xml"

# Création de la cascade et initialisation a la casade de face
faceCascade = cv2.CascadeClassifier(cascPath)

#video
video_capture = cv2.VideoCapture(0)	

# lecture de l'image et convertion en niveau de gris
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()