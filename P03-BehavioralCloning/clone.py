import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = .2
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                current_path = './' + train_dir_name + '/IMG/'
                img_center = cv2.imread(current_path + batch_sample[0].split('/')[-1].lstrip())
                img_left = cv2.imread(current_path + batch_sample[1].split('/')[-1].lstrip())
                img_right = cv2.imread(current_path + batch_sample[2].split('/')[-1].lstrip())
                img_center_flipped = cv2.flip(img_center, 1)
                img_left_flipped = cv2.flip(img_left, 1)
                img_right_flipped = cv2.flip(img_right, 1)

                images.append(img_center)
                images.append(img_left)
                images.append(img_right)
                images.append(img_center_flipped)
                images.append(img_left_flipped)
                images.append(img_right_flipped)
                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)
                angles.append(-steering_center)
                angles.append(-steering_left)
                angles.append(-steering_right)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

lines = []
train_dir_name = 'my_data'
with open('./' + train_dir_name + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=.2)

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

input_shape = (160, 320, 3)

model = Sequential()
model.add(Cropping2D(cropping=((55,25), (0,0)), input_shape=input_shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2,2), activation='relu'))
model.add(Conv2D(nb_filter=36, nb_row=5, nb_col=5, subsample=(2,2), activation='relu'))
model.add(Conv2D(nb_filter=48, nb_row=5, nb_col=5, subsample=(2,2), activation='relu'))
model.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu'))
model.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                                     validation_data=validation_generator, \
                                     nb_val_samples=len(validation_samples), nb_epoch=2)
model.save('model.nvidia.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
