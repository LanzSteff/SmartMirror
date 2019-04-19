import cv2
import numpy as np
# mulitprocessing to open frames in child process to be able to close it
# import multiprocessing
import csv
import shutil
from tkinter import *
from PIL import Image, ImageTk
from tempfile import NamedTemporaryFile

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
age_net = cv2.dnn.readNetFromCaffe(
        'age_gender_model/deploy_age.prototxt',
        'age_gender_model/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe(
        'age_gender_model/deploy_gender.prototxt',
        'age_gender_model/gender_net.caffemodel')
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
gender_list = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX
(width, height) = (600, 600)
csv_file_name = 'data.csv'
# e = multiprocessing.Event()
# p = None


# def show_frame(e):
def show_frame():
    ret, camera_frame = camera.read()  # get camera stream for camera display
    if not ret:
        close()
    # if e.is_set():
    #     camera.release()
    #     cv2.destroyAllWindows()
    #     e.clear()

    camera_frame = cv2.resize(camera_frame, (width, height))
    camera_frame = cv2.flip(camera_frame, 1)  # mirrow mode
    cv2_camera_image = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
    info_frame = np.zeros((height, width, 3), np.uint8)  # create zero screen for info display

    faces = face_cascade.detectMultiScale(cv2_camera_image, 1.1, 5)
    print("Found " + str(len(faces)) + " face(s)")
    # Draw a rectangle around every found face
    for (x, y, w, h) in faces:
        camera_frame, gender, age = get_camera_frame(camera_frame, x, y, w, h)
        info_frame = get_info_frame(info_frame, gender, age)

    cv2_camera_image = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGBA)
    camera_img = Image.fromarray(cv2_camera_image)
    camera_imgtk = ImageTk.PhotoImage(master=camera_display, image=camera_img)
    camera_display.imgtk = camera_imgtk  # Shows frame for display 1
    camera_display.configure(image=camera_imgtk)

    cv2_info_image = cv2.cvtColor(info_frame, cv2.COLOR_BGR2RGBA)
    info_img = Image.fromarray(cv2_info_image)
    info_imgtk = ImageTk.PhotoImage(master=info_display, image=info_img)
    info_display.imgtk2 = info_imgtk  # Shows frame for display 2
    info_display.configure(image=info_imgtk)

    key = cv2.waitKey(1) & 0xFF
    # if the `q` or `ESC` key was pressed, break from the loop
    if key == ord("q") or key == 27:
        close()

    window.after(10, show_frame)


def get_camera_frame(camera_frame, x, y, w, h):
    cv2.rectangle(camera_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    face_img = camera_frame[y:y + h, x:x + w].copy()
    blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]
    overlay_text = "%s, %s" % (gender, age)
    cv2.putText(camera_frame, overlay_text, (x, y), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    return camera_frame, gender, age


def get_info_frame(info_frame, gender, age):
    # Info screen
    # cv2.rectangle(infoFrame, (x, y), (x + w, y + h), (0, 0, 0), 2)
    gender_text = "Gender: %s" % gender
    age_text = "Age: %s" % age
    cv2.putText(info_frame, gender_text, (10, 100), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(info_frame, age_text, (10, 200), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    img = cv2.imread('images/' + gender + age + '.png')
    if img is None:
        img = cv2.imread('images/default.png')
    rows, cols, channels = img.shape
    img = cv2.addWeighted(info_frame[250:250 + rows, 0:0 + cols], 0.5, img, 0.5, 0)
    info_frame[250:250 + rows, 0:0 + cols] = img

    # write_data(gender, age)
    amount_gender, amount_age, amount_gender_age = read_data(gender, age)

    return info_frame


def read_data(gender, age):
    with open(csv_file_name, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            amount_gender = row[gender]
            amount_age = row[age]
            amount_gender_age = row[gender+age]
            return amount_gender, amount_age, amount_gender_age
        return 0, 0, 0


def write_data(gender, age):  # not yet working
    tempfile = NamedTemporaryFile(mode='w', delete=False)
    with open(csv_file_name, 'r') as csvfile, tempfile:
        reader = csv.DictReader(csvfile)
        writer = csv.DictWriter(tempfile, fieldnames=reader.fieldnames)
        for row in reader:
            row[gender] = int(row[gender]) + 1
            row[age] = int(row[age]) + 1
            row[gender+age] = int(row[gender+age]) + 1
            writer.writerow(row)
    shutil.move(tempfile.name, csv_file_name)


def close():  # not yet working
    print('Closing...')
    # e.set()
    # p.join()
    camera.release()
    cv2.destroyAllWindows()
    window.quit()
    window.destroy()


# def show_frame_proc():
#     global p
#     p = multiprocessing.Process(target=show_frame, args=(e,))
#     p.start()


if __name__ == "__main__":
    window = Tk()  # Makes main window
    window.overrideredirect(True)
    window.wm_attributes("-topmost", True)
    window.geometry("+100+100")
    camera_display = Label(window)
    camera_display.grid(row=1, column=0, padx=0, pady=0)  # Camera display
    info_display = Label(window)
    info_display.grid(row=1, column=1, padx=0, pady=0)  # Info display next to camera display

    camera = cv2.VideoCapture(0)  # Laptop webcam
    if camera.isOpened():
        # show_frame_proc()
        show_frame()
        window.mainloop()
