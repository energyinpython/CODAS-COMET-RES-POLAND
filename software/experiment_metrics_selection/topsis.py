import numpy as np
import sys

from additions import *

def TOPSIS(matrix, weights, types, method_distance):
    # Normalize matrix using chosen normalization (for example linear normalization)
    nmatrix = linear_normalization(matrix, types)

    # Multiplicate all rows of normalized matrix by weights
    weighted_matrix = nmatrix * weights

    # Calculate vectors of PIS (ideal solution) and NIS (anti-ideal solution)
    pis = np.max(weighted_matrix, axis=0)
    nis = np.min(weighted_matrix, axis=0)

    # Calculate chosen distance of every alternative from PIS and NIS
    Dp = np.zeros(np.shape(weighted_matrix)[0])
    Dm = np.zeros(np.shape(weighted_matrix)[0])

    for i in range(len(Dp)):
        Dp[i] = method_distance(weighted_matrix[i], pis)
        Dm[i] = method_distance(weighted_matrix[i], nis)

    return Dm / (Dm + Dp)
