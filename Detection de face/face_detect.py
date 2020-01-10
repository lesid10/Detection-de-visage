import cv2
import sys
cv2.destroyAllWindows()
# Obtenir les valeurs fournies par l'utilisateur(nom de l'image et la cascade)
imagePath = sys.argv[1] 
cascPath = "haarcascade_frontalface_default.xml"

# Création de la cascade et initialisation a la casade de face
faceCascade = cv2.CascadeClassifier(cascPath)

# lecture de l'image et convertion en niveau de gris
image = cv2.imread(imagePath)
#gray = cv2.imread(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detection d'image dans le visage
#La detectMultiScalefonction est une fonction qui detecte les objects nous l'appelons sur cascade facial
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2, #Facteur d'echelle
    minNeighbors=5, #fenetre mobile
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
) 

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
