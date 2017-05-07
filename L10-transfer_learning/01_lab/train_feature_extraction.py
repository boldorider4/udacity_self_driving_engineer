import pickle
import os.path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import numpy as np

# Load traffic signs data.
with open('train.p', mode='rb') as f:
    dataset = pickle.load(f)

nb_classes = len(np.unique(dataset['labels']))

# Split data into training and validation sets
train_features, valid_features, train_labels, valid_labels = \
    train_test_split(dataset['features'], dataset['labels'], test_size=0.33, random_state=0)
trainset_size = train_features.shape[0]
validset_size = valid_features.shape[0]

# Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x, (227, 227))
one_hot_y = tf.one_hot(y, nb_classes)

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
# fc8
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=.1))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

# Define loss, training, accuracy operations.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)

# Optimizer (Adam Optimizer)
learning_rate = .01
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])

# Accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train and evaluate the feature extraction model.
epochs = 10
batch_size = 128
divider = 1
actual_trainset_size = int(trainset_size/(divider*batch_size))*batch_size
actual_validset_size = int(validset_size/(divider*batch_size))*batch_size

saver = tf.train.Saver()

def evaluate(X_data, y_data, num_examples, tfsess):
    total_accuracy = 0
    total_loss = 0
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy, loss = tfsess.run((accuracy_operation, loss_operation), feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return (total_accuracy/num_examples, total_loss/num_examples)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    if os.path.isfile('./alexnet_optim.meta'):
        saver.restore(sess, "./alexnet_optim")
        print()
        print("Model restored")
    else:
        print()
        print("Initializing model")

    print("Training...")
    print()

    for i in range(epochs):
        train_features, train_labels = shuffle(train_features, train_labels)
        for offset in range(0, actual_trainset_size, batch_size):
            end = offset + batch_size
            train_batch_x, train_batch_y = train_features[offset:end], train_labels[offset:end]
            sess.run(training_operation, feed_dict={x: train_batch_x, y: train_batch_y})
            
        training_accuracy, training_loss = evaluate(train_features, train_labels, actual_trainset_size, sess)
        validation_accuracy, validation_loss = evaluate(valid_features, valid_labels, actual_validset_size, sess)

        print("Epoch {} ...".format(i+1))
        print("Training Loss = {:.3f}".format(training_loss))
        print("Validation Loss = {:.3f}".format(validation_loss))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './alexnet_optim')
    print("Model saved")
 
