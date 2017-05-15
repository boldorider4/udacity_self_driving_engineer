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

    steering_center = float(line[3])
    # create adjusted steering measurements for the side camera images
    correction = .2
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras
    current_path = './' + train_dir_name + '/'
    img_center = cv2.imread(current_path + line[0].lstrip())
    img_left = cv2.imread(current_path + line[1].lstrip())
    img_right = cv2.imread(current_path + line[2].lstrip())
    img_center_flipped = cv2.flip(img_center, 1)
    img_left_flipped = cv2.flip(img_left, 1)
    img_right_flipped = cv2.flip(img_right, 1)

    images.append(img_center)
    images.append(img_left)
    images.append(img_right)
    images.append(img_center_flipped)
    images.append(img_left_flipped)
    images.append(img_right_flipped)
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)
    measurements.append(-steering_center)
    measurements.append(-steering_left)
    measurements.append(-steering_right)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((55,25), (0,0)), input_shape=X_train.shape[1:]))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(nb_filter=6, nb_row=5, nb_col=5, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(nb_filter=6, nb_row=5, nb_col=5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, nb_epoch=2, validation_split=.2, shuffle=True)
model.save('model.h5')
