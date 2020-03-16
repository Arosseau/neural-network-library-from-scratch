import numpy as np

def cross_entropy_loss(prediction, ground_truth):
    return - np.log(np.dot(prediction, ground_truth))


def dz_cross_entropy_loss(prediction, ground_truth):
    # prediction should be a softmax activation!
    return prediction - ground_truth