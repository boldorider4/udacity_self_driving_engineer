import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # Compute and return softmax(x)
    denom = sum(np.exp(x))
    return [ np.exp(xi)/denom for xi in x ]

logits = [3.0, 1.0, 0.2]
print(softmax(logits))
