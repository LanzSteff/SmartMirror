import cv2
import sys

# cascPath = sys.argv[1]
cascPath = 'C:\\Users\\LanzSteff\\AppData\\Roaming\\Python\\Python37\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
(width, height) = (130, 100)
webcam = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = webcam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    # )
    faces = faceCascade.detectMultiScale(gray, 1.1, 5)


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        # face = gray[y:y + h, x:x + w]
        # face_resize = cv2.resize(face, (width, height))

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    if not ret:
        break

    k = cv2.waitKey(1)
    if k%256 == ord('q') or k%256 == 27:
        print('Closing...')
        break

# When everything is done, release the capture
webcam.release()
cv2.destroyAllWindows()
