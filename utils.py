from scipy.io import loadmat
from datetime import datetime
import os
import cv2


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    name = meta[db][0, 0]["name"][0]
#    celeb_names = meta[db][0, 0]["celeb_names"][0]
#    celeb_id = meta[db][0, 0]["celeb_id"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age, name


def load_data(mat_path):
    d = loadmat(mat_path)

    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]


def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass
    
def get_face(imagePath):
    '''
    возвращает массив с самым большим лицом на фото
    если лица нет, возвращает []
    '''
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(50, 50)
    )
    try:
        x_max = faces[0][0]
    except:
        return []
    y_max = faces[0][1]
    w_max = faces[0][2]
    h_max = faces[0][3]
    for (x, y, w, h) in faces:
        if w > w_max:
            x_max = x
            y_max = y
            w_max = w
            h_max = h       
    cropped = image[y_max:y_max+h_max,x_max:x_max+w_max]
    return cropped    