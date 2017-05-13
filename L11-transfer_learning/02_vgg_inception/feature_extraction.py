import pickle
import tensorflow as tf
import numpy as np
# import Keras layers you need here
from keras.layers import Input, Flatten, Dense, Activation
from keras.models import Model
from sklearn.preprocessing import LabelBinarizer

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 30, "Number of epochs to train")
flags.DEFINE_integer('batch_size', 256, "Training batch size")

if int(FLAGS.epochs) < 1:
    print('Number of epochs should be at least 1')
    exit(-1)

if int(FLAGS.batch_size) < 1:
    print('batch_size should be at least 1')
    exit(-1)

def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print('training shape = ', X_train.shape, y_train.shape)
    print('validation shape = ', X_val.shape, y_val.shape)
    print('epochs = ', FLAGS.epochs)
    print('batch_size = ', FLAGS.batch_size)
    
    n_classes = len(np.unique(y_train))
    input_shape       = X_train.shape[1:]
    input_layer       = Input(shape=input_shape)
    flatten_layer     = Flatten()(input_layer)
    densely_connected = Dense(n_classes)(flatten_layer)
    softmax_out       = Activation('softmax')(densely_connected)
    model = Model(input_layer, softmax_out)
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])

    # train your model here
    history = model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, \
                        batch_size=FLAGS.batch_size, validation_data=(X_val,y_val))
    
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
