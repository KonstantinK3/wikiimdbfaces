import cv2
from utils import get_meta, get_face
import numpy as np
from skimage import io
from skimage.transform import resize
from keras.utils.np_utils import to_categorical

db = "imdb"
mat_path = f"data/{db}_crop/{db}.mat"
imgs_path = f"data/{db}_crop/"
data_path = f"data/{db}_arrays_cropped_faces/"
full_path, dob, gender, photo_taken, face_score, second_face_score, age, name = get_meta(mat_path, db)

ending = (len(full_path))
batch_size = 10000
iterations = ending//batch_size+1
image_size = 227

for iteration in range(29, iterations+1):
    current_iteration = iteration * batch_size
    last_iteration = current_iteration + batch_size
    if last_iteration > ending:
        last_iteration = ending
    print (f"current iteration {current_iteration} last_iteration {last_iteration}")
    imgs = []
    gender_labels = []
    age_labels = []
    name_labels = []
    for counter in range(current_iteration, last_iteration):
        print (counter)
        if face_score[counter] > 0.5 and (gender[counter] == 0 or gender[counter] == 1):
            cropped_face = get_face(imgs_path + full_path[counter][0])
            if (len(cropped_face) != 0):
                resized = cv2.resize(cropped_face, (image_size,image_size), interpolation = cv2.INTER_LINEAR)
                gender_labels.append(gender[counter])
                age_labels.append(age[counter])
                name_labels.append(name[counter][0])
                imgs.append(resized)
    img_array = np.array(imgs)
    gender_array = to_categorical(np.array(gender_labels)) #[0, 1] - man, [1, 0] - woman
    age_array = np.array(age_labels)
    print (f'saving arrays, data: {data_path}, iteration: {iteration}')
    np.save(f'{data_path}images_{db}_{iteration}', img_array)
    np.save(f'{data_path}gender_{db}_{iteration}', gender_array)
    np.save(f'{data_path}age_{db}_{iteration}', age_array)
    np.save(f'{data_path}/names_{db}_gender_{iteration}', name_labels)

#counter = 111500
#imagePath = imgs_path + full_path[counter][0]
#image_full = cv2.imread(imagePath)
#cv2.imshow("full", image_full)
#cropped_face = get_face(imagePath)


