# -------------------------------------------------------------------------
'''
    Problem 2: Compute the objective function and decision function of dual SVM.

'''
from problem1 import *

import numpy as np

# -------------------------------------------------------------------------
def dual_objective_function(alpha, train_y, train_X, kernel_function, sigma):
    """
    Compute the dual objective function value.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix. n: number of features; m: number training examples.
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.
    :return: a scalar representing the dual objective function value at alpha
    Hint: refer to subsection 3.2 "Dual problem for SVM" in notes, specifically, Eqn. 55.
          You can try to call kernel_function.__name__ to figure out which kernel are used.
    """
    #########################################
    L_alpha = 0
    if kernel_function.__name__ == "Gaussian_kernel":
        Grim_matrix = kernel_function(train_X, train_X, sigma)
    else:
        Grim_matrix = kernel_function(train_X, train_X)
    for (i, i_th_element) in enumerate(alpha[0]):
        L_alpha = L_alpha + i_th_element
        for (j, j_th_element) in enumerate(alpha[0]):
            L_alpha = L_alpha - (1 / 2 * i_th_element * j_th_element * train_y[i] * train_y[j] * Grim_matrix[i][j])
    return L_alpha
    #########################################


# -------------------------------------------------------------------------
def primal_objective_function(alpha, train_y, train_X, b, C, kernel_function, sigma):
    """
    Compute the primal objective function value.
    When with linear kernel:
        The primal parameter w is recovered from the dual variable alpha.
    When with Gaussian kernel:
        Can't recover the primal parameter and kernel trick needs to be used to compute the primal objective function.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x C: regularization parameter of soft-SVM
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.

    :return: a scalar representing the primal objective function value at alpha
    Hint: rem training feature matrix.
    b: bias term
    C: regularization parameter of soft-SVM
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.

    :return: a scalar representing the primal objective function value at alpha
    Hint: refer to subsection 3.4 "Non-separabale problems" in notes, specifically, Eqn. 62-64.
    """
    #########################################
    L_omega = 0
    if kernel_function.__name__ == "Gaussian_kernel":
        Grim_matrix = kernel_function(train_X, train_X, sigma)
    else:
        Grim_matrix = kernel_function(train_X, train_X)
    for (i, i_th_element) in enumerate(alpha[0]):
        # if i_th_element == C:
        #     L_omega = L_omega + C - (C * train_y[i] * b)
        #     for (j_, j_th_element_) in enumerate(alpha[0]):
        #         L_omega = L_omega - (C * train_y[i] * j_th_element_ * train_y[j_] * Grim_matrix[i][j_])
        xi = 0 # ξ
        for (j, j_th_element) in enumerate(alpha[0]):
            L_omega = L_omega + (1 / 2 * i_th_element * j_th_element * train_y[i] * train_y[j] * Grim_matrix[i][j])
            xi = xi + (train_y[i] * j_th_element * train_y[j] * Grim_matrix[i][j])
        xi = 1 - xi - b * train_y[i]
        if xi > 0:
            L_omega = L_omega + C * xi
    return L_omega
    #########################################



def decision_function(alpha, train_y, train_X, b, kernel_function, sigma, test_X):
    """
    Compute the linear function <w, x> + b on examples in test_X, using the current SVM.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    test_X: n x m2 test feature matrix.
    b: scalar, the bias term in SVM <w, x> + b.
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.

    :return: 1 x m2 vector <w, x> + b
    """
    #########################################
    decision_vector = np.zeros((test_X[0].size, ))
    if kernel_function.__name__ == "Gaussian_kernel":
        Grim_matrix = kernel_function(test_X, train_X, sigma)
    else:
        Grim_matrix = kernel_function(test_X, train_X)
    for (i, i_th_element) in enumerate(Grim_matrix):
        f_x = 0
        for (j, j_th_element) in enumerate(alpha[0]):
            f_x = f_x + j_th_element * train_y[0, j] * i_th_element[j]
        f_x = f_x + b
        decision_vector[i] = f_x
    return decision_vector
    #########################################
