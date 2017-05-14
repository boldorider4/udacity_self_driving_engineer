import csv
import cv2
import numpy as np

lines = []
train_dir_name = 'data'
with open('./' + train_dir_name + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './' + train_dir_name + '/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=X_train.shape[1:]))
model.add(Conv2D(nb_filter=6, nb_row=5, nb_col=5, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(nb_filter=6, nb_row=5, nb_col=5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, nb_epoch=5, validation_split=.2, shuffle=True)
model.save('model.h5')
