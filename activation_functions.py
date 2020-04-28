import numpy as np

'''
We only calculate the partial derivatives to the same input variable,
since dz_a will often bring about a Kronecker delta (except for softmax, 
but there we immediately go to dz_loss_function) and therefore we
don't need the entire partial derivative matrix (aka. the Jacobian), only
the diagonal elements which we'll then use in a Hadamard product.
'''

activation_fs = {
    "identity": "identity",
    "relu": "ReLU_activation",
    "softmax": "softmax_activation"
}


def identity(z):
    return z


def dz_identity(z):
    return np.ones(z.shape)


def ReLU_activation(z):
    z[z < 0] = 0
    return z
    # return np.maximum(z, np.zeros(z.shape))


def dz_ReLU_activation(z):
    dz_relu = np.zeros(z.shape)
    dz_relu[z > 0] = 1  # return 1 where z > 0 else 0
    return dz_relu


def softmax_activation(z):
    exponentials = np.exp(z)
    # divide every i-th row of the 'exponentials' matrix by the i-th element of the sum vector:
    return exponentials / np.sum(exponentials, axis=1)[:, None]
