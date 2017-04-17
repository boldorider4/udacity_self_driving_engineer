import math
import numpy as np

def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    n_batches = math.ceil(len(features)/batch_size)
    # Implement batching
    output = []
    for b in range(0, n_batches-1):
        feature_batch = features[(b*batch_size):((b+1) * batch_size - 1), :]
        labels_batch = labels[(b*batch_size):((b+1) * batch_size - 1), :]
        output.append([feature_batch, labels_batch])

    feature_batch = features[((n_batches-1)*batch_size):(len(features)-1), :]
    labels_batch = labels[((n_batches-1)*batch_size):(len(labels)-1), :]
    output.append([feature_batch, labels_batch])
