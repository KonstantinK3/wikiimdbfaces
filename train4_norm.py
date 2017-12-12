
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
import numpy as np

arrays_number = 10 #количество массивов. 8 масс по 10 эпох - кул
epochs_number = 5
db = "wiki"
data_path = f"data/{db}_arrays/"

model_number = 'test'

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', strides=3, input_shape=(150, 150, 3)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', strides=3, input_shape=(150, 150, 3)))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu', strides=3, input_shape=(150, 150, 3)))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), activation='relu', strides=3, input_shape=(150, 150, 3)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

for array_num in range(0, arrays_number+1):
    print(f'loading {data_path}X_{db}_gender_{array_num}.npy')
    X = np.load(f'{data_path}X_{db}_gender_{array_num}.npy')
    print(f'loading {data_path}/y_{db}_gender_{array_num}.npy')
    y = np.load(f'{data_path}/y_{db}_gender_{array_num}.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model.fit(X_train, y_train, 
              epochs=epochs_number,
              batch_size=100,
              validation_data=(X_test, y_test))

print (f'saving model models/{db}_model_gender_{model_number}.h5')
model.save(f'models/{db}_model_gender_{model_number}.h5')









