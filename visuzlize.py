from __future__ import print_function
from keras.models import load_model
import cv2
from utils import get_face
from scipy.misc import imsave
import numpy as np
import time
from keras import backend as K

# dimensions of the generated pictures for each filter.
img_width = 227 
img_height = 227

model_number = 'convBatch'
classifier = load_model(f'models/imdb_model_gender_{model_number}.h5')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
# the name of the layer we want to visualize

layer_name = 'conv2d_12' #conv2d_13
filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

# build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:, :, :, filter_index])

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])

# we start from a gray image with some noise
input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128.
# run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step
    
    
    
    