import matplotlib.pyplot as plt
import numpy as np
from utils import get_meta
import matplotlib.image as mpimg
from skimage import io
from skimage.transform import rescale, resize

img = mpimg.imread('images/f/1.jpg')
img = resize(img, (150,150,3))

plt.imshow(img)
plt.show()