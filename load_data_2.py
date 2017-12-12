import matplotlib.pyplot as plt
import numpy as np
from utils import get_meta
from skimage import io
from skimage.transform import resize
from keras.utils.np_utils import to_categorical



db = "wiki"
# db = "imdb"
mat_path = f"data/{db}_crop/{db}.mat"
imgs_path = f"data/{db}_crop/"
data_path = f"data1/{db}_arrays/"
full_path, dob, gender, photo_taken, face_score, second_face_score, age, name = get_meta(mat_path, db)

ending = (len(full_path))
batch_size = 2000
iterations = ending//batch_size+1

for iteration in range(0, iterations+1):
    current_iteration = iteration * batch_size
    last_iteration = current_iteration + batch_size
    if last_iteration > ending:
        last_iteration = ending
    print (f"current iteration {current_iteration} last_iteration {last_iteration}")
    imgs = []
    labels = []
    for counter in range(current_iteration, last_iteration):
        print (counter)
        if face_score[counter] > 0.5 and (gender[counter] == 0 or gender[counter] == 1):
            labels.append(gender[counter])
            img = io.imread(imgs_path + full_path[counter][0])
            img = resize(img, (150,150,3))
            imgs.append(img)
    X = np.array(imgs)
    y = to_categorical(np.array(labels)) #[0, 1] - man, [1, 0] - woman
    print (f'saving arrays, data: {data_path}X_wiki_gender_{iteration}')
    np.save(f'{data_path}X_{db}_gender_{iteration}', X)
    np.save(f'{data_path}/y_{db}_gender_{iteration}', y)
            


#plt.imshow(X[3])
#Y[1]




