import cv2, time, numpy as np

# initialize camera
cascPath = 'C:\\Users\\LanzSteff\\AppData\\Roaming\\Python\\Python37\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
(width, height) = (600, 1200)
font = cv2.FONT_HERSHEY_SIMPLEX
webcam = cv2.VideoCapture(0)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
genderList = ['Male', 'Female']

# allow the camera to warmup
time.sleep(0.1)

def initializeCaffeModel():
    print('Loading models...')
    ageNet = cv2.dnn.readNetFromCaffe(
        'age_gender_model/deploy_age.prototxt',
        'age_gender_model/age_net.caffemodel')
    genderNet = cv2.dnn.readNetFromCaffe(
        'age_gender_model/deploy_gender.prototxt',
        'age_gender_model/gender_net.caffemodel')
    return (ageNet, genderNet)


def captureLoop(ageNet, genderNet):
    # capture frames from the camera
    while True:
        # Capture frame-by-frame
        ret, frame = webcam.read()
        frame = cv2.flip(frame, 1)
        infoFrame = np.zeros((width, height, 3), np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 5)
        print("Found " + str(len(faces)) + " face(s)")

        # Draw a rectangle around every found face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            faceImg = frame[y:y + h, x:x + w].copy()
            blob = cv2.dnn.blobFromImage(faceImg, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # Predict gender
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            # Predict age
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            overlayText = "%s, %s" % (gender, age)
            cv2.putText(frame, overlayText, (x, y), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
            infoFrame = getInfoScreen(infoFrame, gender, age)

        cv2.imshow("Image", frame)
        cv2.imshow("Info", infoFrame)
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        # if the `q` or `ESC` key was pressed, break from the loop
        if key == ord("q") or key == 27:
            print('Closing...')
            break

def getInfoScreen(infoFrame, gender, age):
    # Info screen
    # cv2.rectangle(infoFrame, (x, y), (x + w, y + h), (0, 0, 0), 2)
    genderText = "Gender: %s" % gender
    ageText = "Age: %s" % age
    cv2.putText(infoFrame, genderText, (10, 100), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(infoFrame, ageText, (10, 200), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    img = cv2.imread('images/' + gender + age + '.png')
    if img is None:
        img = cv2.imread('images/default.png')
    rows, cols, channels = img.shape
    img = cv2.addWeighted(infoFrame[250:250 + rows, 0:0 + cols], 0.5, img, 0.5, 0)
    infoFrame[250:250 + rows, 0:0 + cols] = img

    return infoFrame


if __name__ == '__main__':
    ageNet, genderNet = initializeCaffeModel()
    captureLoop(ageNet, genderNet)
    # When everything is done, release the capture
    webcam.release()
    cv2.destroyAllWindows()
