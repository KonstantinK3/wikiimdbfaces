
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import RMSprop
import numpy as np

arrays_number = 0 #количество массивов.
epochs_number = 20
db = "imdb"
data_path = f"data/{db}_arrays_cropped_faces/"

model_number = '1'

model = Sequential()
model.add(Conv2D(96, (7, 7), strides=4, input_shape=(227, 227, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization())

model.add(Conv2D(256, (5, 5), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

rms = RMSprop(lr = 0.002)

model.compile(loss='categorical_crossentropy',
              optimizer=rms,
              metrics=['accuracy'])

for array_num in range(0, arrays_number+1):
    print(f'loading {data_path}images_{db}_{array_num}.npy')
    X = np.load(f'{data_path}images_{db}_{array_num}.npy')
    print(f'loading {data_path}gender_{db}_{array_num}.npy')
    y = np.load(f'{data_path}gender_{db}_{array_num}.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train, 
              epochs=epochs_number,
              batch_size=50,
              validation_data=(X_test, y_test))

#--------
#array_num = 0
#print(f'loading {data_path}images_{db}_{array_num}.npy')
#X = np.load(f'{data_path}images_{db}_{array_num}.npy')
#
#im_num = 2
#cv2.imshow(f'{im_num}', X[im_num])









