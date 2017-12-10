import numpy as np
from keras.preprocessing import image
from keras.models import load_model

model_number = 3
classifier = load_model(f'models/wiki_model_gender_{model_number}.h5')

def gender(image_num, cat):
    image_name = f"images/{cat}/{image_num}.jpg"
    test_image = image.load_img(image_name, target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predict = classifier.predict(test_image)
#    print (predict)
    if predict[0,0] == 0:
        return 'man'
    else:
        return 'woman'

for i in range(1,9):
    print (f"image {i} is a " +  gender(i, 'f'))