import numpy as np


loss_fs = {
            "quadratic": "quadratic_loss",
            "cross_entropy": "cross_entropy_loss"
        }


def cross_entropy_loss(prediction, ground_truth):
    # return - np.log(np.dot(prediction, ground_truth))
    loss = - np.sum(np.multiply(ground_truth, np.log(prediction)), axis=1)
    return loss


def dz_cross_entropy_loss(prediction, ground_truth):
    # prediction should be a softmax activation!
    return prediction - ground_truth


def quadratic_loss(prediction, ground_truth):
    diff = prediction - ground_truth
    return 0.5 * np.sum(np.multiply(diff, diff), axis=1)


def da_quadratic_loss(prediction, ground_truth):
    return - 1
