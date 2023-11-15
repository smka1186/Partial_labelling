import numpy as np

def calAccuracy(test_outputs, test_target):
    p, q = test_target.shape
    idx1 = np. argmax(test_outputs, axis=1)
    idx2 = np. argmax(test_target, axis=1)
    accuracy = (np.sum(idx1 == idx2))/p
    return accuracy

