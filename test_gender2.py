import numpy as np
from keras.preprocessing import image
from keras.models import load_model
classifier = load_model('model1.h5')

def gender(image_name):
    image_name = 'images/f/' + image_name
    test_image = image.load_img(image_name, target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predict = classifier.predict(test_image)
    print (predict)
    if predict[0,0] == 1:
        return 'man'
    else:
        return 'woman'

for i in range(1,8):
    print ("image {} is a {}".format(i, gender(str(i)+'.jpg')))