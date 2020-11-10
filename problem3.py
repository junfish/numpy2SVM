# -------------------------------------------------------------------------
'''
    Problem 3: SMO training algorithm

'''
from problem1 import *
from problem2 import *

import numpy as np

import copy

class SVMModel():
    """
    The class containing information about the SVM model, including parameters, data, and hyperparameters.

    DONT CHANGE THIS DEFINITION!
    """
    def __init__(self, train_X, train_y, C, kernel_function, sigma=1):
        """
            train_y: 1 x m labels (-1 or 1) of training data.
            train_X: n x m training feature matrix. n: number of features; m: number training examples.
            C: a positive scalar
            kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
            sigma: need to be provided when Gaussian kernel is used.
        """
        # data
        self.train_X = train_X
        self.train_y = train_y
        self.n, self.m = train_X.shape

        # hyper-parameters
        self.C = C
        self.kernel_func = kernel_function
        self.sigma = sigma

        # parameters
        self.alpha = np.zeros((1, self.m))
        self.b = 0

def train(model, max_iters = 10, record_every = 1, max_passes = 1, tol=1e-6):
    """
    SMO training of SVM
    model: an SVMModel
    max_iters: how many iterations of optimization
    record_every: record intermediate dual and primal objective values and models every record_every iterations
    max_passes: each iteration can have maximally max_passes without change any alpha, used in the SMO alpha selection.
    tol: numerical tolerance (exact equality of two floating numbers may be impossible).
    :return: 4 lists (of iteration numbers, dual objectives, primal objectives, and models)
    Hint: refer to subsection 3.5 "SMO" in notes.
    """
    #########################################
    if model.kernel_func.__name__ == "Gaussian_kernel":
        Grim_matrix = model.kernel_func(model.train_X, model.train_X, model.sigma)
    else:
        Grim_matrix = model.kernel_func(model.train_X, model.train_X)
    iteration_numbers = []
    primal_objectives = []
    dual_objectives = []
    models = []
    for i in range(max_iters):
        num_passes = 0
        while(num_passes < max_passes):
            num_changes = 0
            for j in range(model.m):
                j_margin = 0 # compute margin of j, which is also g_j
                for k in range(model.m):
                    j_margin = j_margin + model.train_y[0, j] * model.alpha[0, k] * model.train_y[0, k] * Grim_matrix[j, k]
                j_margin = j_margin + model.train_y[0, j] * model.b

                KKT_condition =  ((model.alpha[0, j] == 0) and (j_margin >= 1)) or \
                                 ((model.alpha[0, j] > 0) and (model.alpha[0, j] < model.C) and (j_margin == 1)) or \
                                 ((model.alpha[0, j] == model.C) and (j_margin < 1))

                if not KKT_condition:
                    while True:
                        random_index = np.random.randint(0, model.m)
                        if (random_index != j):
                            break
                    # Compute L&H
                    rho = 0
                    for k in range(model.m):
                        if (k != j) and (k != random_index):
                            rho = rho - model.alpha[0, k] * model.train_y[0, k]
                    if model.train_y[0, j] * model.train_y[0, random_index] < 0: # [-C, C]
                        if model.train_y[0, j] * rho < 0:
                            L = - model.train_y[0, j] * rho
                            H = model.C
                        else:
                            L = 0
                            H = model.C - model.train_y[0, j] * rho
                    else: # [0, 2C]
                        if model.train_y[0, j] * rho < model.C:
                            L = 0
                            H = model.train_y[0, j] * rho
                        else:
                            L = model.train_y[0, j] * rho - model.C
                            H = model.C
                    # compute g_j, g_random_index
                    g_j = j_margin * model.train_y[0, j]
                    g_random_index = 0  # compute g_random_index
                    for k in range(model.m):
                        g_random_index = g_random_index + model.alpha[0, k] * model.train_y[0, k] * \
                                   Grim_matrix[random_index, k]
                    g_random_index = g_random_index + model.b
                    alpha_random_index = model.alpha[0, random_index] + \
                                         model.train_y[0, random_index] * (g_j - g_random_index - model.train_y[0, j] + model.train_y[0, random_index]) \
                                         / (Grim_matrix[j, j] + Grim_matrix[random_index, random_index] - 2 * Grim_matrix[j, random_index])
                    alpha_random_index_old = model.alpha[0, random_index]
                    if alpha_random_index > H:
                        model.alpha[0, random_index] = H
                    if alpha_random_index < L:
                        model.alpha[0, random_index] = L
                    else:
                        model.alpha[0, random_index] = alpha_random_index
                    if np.allclose(model.alpha[0, random_index], alpha_random_index_old, atol = 0, rtol = tol):
                        model.alpha[0, random_index] = alpha_random_index_old
                        break
                    model.alpha[0, j] = model.alpha[0, j] + model.train_y[0, j] * model.train_y[0, random_index] * \
                                        (alpha_random_index_old - model.alpha[0, random_index])
                    support_vector_index = 0
                    for k in range(model.m):
                        if (model.alpha[0, k] < model.C and model.alpha[0, k] > 0):
                            support_vector_index = k
                            break
                    new_b = model.train_y[0, support_vector_index]
                    for k in range(model.m):
                        new_b = new_b - model.alpha[0, k] * model.train_y[0, k] * Grim_matrix[k, support_vector_index]
                    model.b = new_b
                    num_changes = num_changes + 1
            if num_changes == 0:
                num_passes = num_passes + 1
            else:
                num_passes = 0
        if i % record_every == 0:
            iteration_numbers.append(i)
            primal_objectives.append(primal_objective_function(model.alpha, model.train_y[0], model.train_X, model.b, model.C, model.kernel_func, model.sigma))
            dual_objectives.append(dual_objective_function(model.alpha, model.train_y[0], model.train_X, model.kernel_func, model.sigma))
            models.append(model)
    return iteration_numbers, primal_objectives, dual_objectives, models

    #########################################

def predict(model, test_X):
    """
    Predict the labels of test_X
    model: an SVMModel
    test_X: n x m matrix, test feature vectors
    :return: 1 x m matrix, predicted labels
    """
    #########################################
    predict = []
    if model.kernel_func.__name__ == "Gaussian_kernel":
        Grim_matrix = model.kernel_func(test_X, model.train_X, model.sigma)
    else:
        Grim_matrix = model.kernel_func(test_X, model.train_X)
    for (x_index, x) in enumerate(test_X.T):
        predict_value = 0
        for (alpha_index, alpha_element) in enumerate(model.alpha[0]):
            predict_value += alpha_element * model.train_y[0, alpha_index] * Grim_matrix[x_index, alpha_index]
        predict_value += model.b
        if predict_value > 0:
            predict_value = 1
        elif predict_value < 0:
            predict_value = -1
        predict.append(predict_value)
    print(predict)
    return predict
    #########################################
