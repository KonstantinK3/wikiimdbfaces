import numpy as np
from keras.models import load_model
import cv2
from utils import get_face

model_number = 'convBatch'
classifier = load_model(f'models/imdb_model_gender_{model_number}.h5')

def gender(image_num, cat):
    image_name = f"images/{cat}/{image_num}.jpg"
    cropped_face = get_face(image_name)
    resized = cv2.resize(cropped_face, (227,227), interpolation = cv2.INTER_LINEAR)
#    cv2.imshow('image_num', cropped_face)
    test_image = np.expand_dims(resized, axis = 0)
    predict = classifier.predict(test_image)
#    print (predict, predict[0,0], predict[0,1])
    if predict[0,0] > predict[0,1]:
        detected_gender = 'woman'
    else:
        detected_gender = 'man'
    return detected_gender, round(predict[0,0]*100,4), round(predict[0,1]*100,4)
#    return detected_gender, predict[0,0], predict[0,1]

for i in range(1,9):
    detected_gender, woman_score, man_score = gender(i, 'f')
    print (f"image {i} is a {detected_gender}, score is {woman_score} / {man_score}")