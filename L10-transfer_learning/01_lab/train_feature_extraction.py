import pickle
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
dataset_size = dataset['features'].shape[0]
train_size = int(dataset_size*.9)
train_features, train_labels = dataset['features'][:train_size], dataset['labels'][:train_size]
trainset_size = train_features.shape[0]
valid_features, valid_labels = dataset['features'][train_size:], dataset['labels'][train_size:]
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
# fc8, 1000
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
training_operation = optimizer.minimize(loss_operation)

# Accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train and evaluate the feature extraction model.
epochs = 5
batch_size = 512

def evaluate(X_data, y_data, num_examples):
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy, loss = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    print("Initializing model")
    print("Training...")
    print()

    for i in range(epochs):
        train_features, train_labels = shuffle(train_features, train_labels)
        for offset in range(0, trainset_size, batch_size):
            end = offset + batch_size
            batch_x, batch_y = train_features[offset:end], train_labels[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        training_accuracy, training_loss = evaluate(train_features, train_labels, trainset_size)
        validation_accuracy, validation_loss = evaluate(valid_features, train_labels, validset_size)

        print("Epoch {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
 
