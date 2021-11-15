import numpy as np
import sys

from additions import *

# CODAS method
def CODAS(matrix, weights, types, method_distance, tau_v):
    # Normalize matrix using linear normalization
    nmatrix = linear_normalization(matrix, types)
    
    # Multiply all rows of normalized matrix by weights
    weighted_matrix = nmatrix * weights
    m, n = weighted_matrix.shape

    # Calculate NIS vector (anti-ideal solution)
    nis = np.min(weighted_matrix, axis=0)

    # Calculate chosen distance (for example Euclidean) and Taxicab distance from anti-ideal solution

    # Calculate chosen distance
    E = np.zeros(np.shape(weighted_matrix)[0])
    
    for i in range(len(E)):
        E[i] = method_distance(weighted_matrix[i], nis)

    # Calculate Taxicab (Manhattan) distance
    T = np.sum(np.abs(weighted_matrix - nis), axis=1)
    
    # Construct the relative assessment matrix
    h = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            h[i, j] = (E[i] - E[j]) + (psi(E[i] - E[j], tau_v) * (T[i] - T[j]))

    return np.sum(h, axis=1)

