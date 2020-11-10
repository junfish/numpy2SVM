'''
    Problem 1: Implement linear and Gaussian kernels and hinge loss
'''

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def linear_kernel(X1, X2):
    """
    Compute linear kernel between two set of feature vectors.
    No constant 1 is appended.

    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    :return: if both m1 and m2 are 1, return linear kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=linear kernel of column i from X1 and column j from X2
    """
    #########################################
    # if X1 and X2 are vector (size = (n,), which are seen as feature vector
    # try:
    #     n, m1 = X1.shape
    # except ValueError:
    #     X1 = X1[:, np.newaxis]
    #     X2 = X2[:, np.newaxis]
    #     n, m1 = X1.shape
    # if (m1 == 1):
    #     return np.dot(X1.T, X2)[0][0]
    # else:
    return np.dot(X1.T, X2)
    #########################################


def Gaussian_kernel(X1, X2, sigma=1):
    """
    Compute linear kernel between two set of feature vectors.
    No constant 1 is appended.
    For your convenience, please use euclidean_distances.

    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    sigma: Gaussian variance (called bandwidth)
    :return: if both m1 and m2 are 1, return Gaussian kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=Gaussian kernel of column i from X1 and column j from X2

    """
    #########################################
    distances = euclidean_distances(X1.T, X2.T, squared = True) # return size: (m1, m2)
    return np.exp(-distances / (2 * sigma ** 2))
    #########################################

def hinge_loss(z, y):
    """
    Compute the hinge loss on a set of training examples
    z: 1 x m vector, each entry is <w, x> + b (may come from a kernel function)
    y: 1 x m label vector. Each entry is -1 or 1
    :return: 1 x m hinge losses over the m examples
    """
    #########################################
    soft_value_vector = 1 - np.multiply(z, y)
    for (index, element) in enumerate(soft_value_vector):
        if element < 0:
            soft_value_vector[index] = 0
    return soft_value_vector
    #########################################