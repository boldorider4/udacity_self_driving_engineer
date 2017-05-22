import csv
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path
from math import ceil

# generator function to yield batches of samples used to train or validate a model
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            # create batch
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = .15
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                current_path = './' + train_dir_name + '/IMG/'
                img_center = mpimg.imread(current_path + batch_sample[0].split('/')[-1].lstrip())
                img_left = mpimg.imread(current_path + batch_sample[1].split('/')[-1].lstrip())
                img_right = mpimg.imread(current_path + batch_sample[2].split('/')[-1].lstrip())
                # mirror images horizontally
                img_center_flipped = np.fliplr(img_center)
                img_left_flipped = np.fliplr(img_left)
                img_right_flipped = np.fliplr(img_right)

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
            #yield batch
            yield sklearn.utils.shuffle(X_train, y_train)

lines = []
# name of directory containing driving_log.csv and IMG directory 
train_dir_name = 'my_data'
# open driving log used to get image filenames and corresponding steering values
with open('./' + train_dir_name + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# split dataset into training and validation set
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=.2)

# number of lines x 3 (one for each center/left/right file) x 2 (flipped images)
nb_train_samples = len(lines) * 3 * 2
nb_validation_samples = ceil(nb_train_samples * .2)
# create batch generators for training and validation sets
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

input_shape = (160, 320, 3)

# model architecture
model = Sequential()
model.add(Cropping2D(cropping=((55,25), (0,0)), input_shape=input_shape, name='input_layer'))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, name='lambda_layer'))
model.add(Conv2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2,2), activation='relu', name='conv2d24_layer'))
model.add(Conv2D(nb_filter=36, nb_row=5, nb_col=5, subsample=(2,2), activation='relu', name='conv2d36_layer'))
model.add(Conv2D(nb_filter=48, nb_row=5, nb_col=5, subsample=(2,2), activation='relu', name='conv2d48_layer'))
model.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', name='conv2d64_layer'))
model.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', name='conv2d64_2_layer'))
model.add(Dropout(.6, name='dropout1_layer'))
model.add(Flatten(name='flatten_layer'))
model.add(Dense(1164, name='dense1164_layer'))
model.add(Dense(100, name='dense100_layer'))
model.add(Dropout(.7, name='dropout2_layer'))
model.add(Dense(50, name='dense50_layer'))
model.add(Dropout(.5, name='dropout3_layer'))
model.add(Dense(1, name='output_layer'))

model.compile(optimizer='adam', loss='mse')
for layer in model.layers[:9]:
    layer.trainable = False

# loading model weights
if os.path.isfile('./model_weights.easy_track.h5'):
    print('loaded weights into model!')
    model.load_weights('model_weights.easy_track.h5', by_name=True)
# if no model weight is present, try and load entire model
elif os.path.isfile('./model.easy_track.h5'):
    print('loaded entire model!')
    model = load_model('model.easy_track.h5')

# change learning rate when fine tune
model.optimizer.lr.assign(0.0005)
# run training session
history_object = model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, \
                                     validation_data=validation_generator, \
                                     nb_val_samples=nb_validation_samples, nb_epoch=5)

# saving the model and weights
print('weights and model saved!')
model.save('model.easy_track.h5')
model.save_weights('model_weights.easy_track.h5')

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
