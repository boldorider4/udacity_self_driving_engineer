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
    # Implement batching
    output = []
    temp_features = []
    temp_labels = []

    for b in range(0, len(features)):
        temp_features.append(features[b])
        temp_labels.append(labels[b])
        if (math.fmod(b, batch_size) == batch_size - 1.0) or b == len(features)-1:
            output.append([temp_features, temp_labels])
            temp_features = []
            temp_labels = []

    return output
