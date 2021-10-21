import numpy as np
import sys
from scipy.stats import pearsonr

###
# euclidean distance
def euclidean_distance(A, B):
    tmp = np.sum(np.square(A - B))
    return np.sqrt(tmp)


# for hausdorff distance
def hausdorff(A, B):
    min_h = np.inf
    for i in range(len(A)):
        for j in range(len(B)):
            d = euclidean_distance(A[i], B[j])
            if d < min_h:
                min_h = d
                min_ind = j

    max_h = -np.inf
    for i in range(len(A)):
        d = euclidean_distance(A[i], B[min_ind])
        if d > max_h:
            max_h = d

    return max_h


# hausdorff distance
def hausdorff_distance(A, B):
    ah = hausdorff(A, B)
    bh = hausdorff(B, A)
    return max(ah, bh)


# correlative distance
def correlative_distance(A, B):
    num = np.sum((A - np.mean(A)) * (B - np.mean(B)))
    denom = np.sqrt(np.sum((A - np.mean(A)) ** 2)) * np.sqrt(np.sum((B - np.mean(B)) ** 2))
    if denom == 0:
        denom = sys.float_info.epsilon
        print('Warning in correlative_distance!')
    return 1 - (num / denom)


# chebyshev distance
def chebyshev_distance(A, B):
    max_h = -np.inf
    for i in range(len(A)):
        for j in range(len(B)):
            d = np.abs(A[i] - B[j])
            if d > max_h:
                max_h = d
    return max_h


# standarized euclidean distance
def std_euclidean_distance(A, B):
    tab_std = np.vstack((A, B))
    stdv = np.std(tab_std, axis = 0)
    stdv[stdv == 0] = sys.float_info.epsilon
    tmp = np.sum(np.square((A - B) / stdv))
    return np.sqrt(tmp)

# cosine distance
def cosine_distance(A, B):
    num = np.sum(A * B)
    denom = (np.sqrt(np.sum(np.square(A)))) * (np.sqrt(np.sum(np.square(B))))
    if denom == 0:
        denom = sys.float_info.epsilon
        print('Warning in cosine_distance!')
    return 1 - (num / denom)


# for CODAS method
# 0.01 - 0.05 recommended range of tau value
def psi(x, tau=0.01):
    if np.abs(x) >= tau:
        return 1
    else:
        return 0


# linear normalization
def linear_normalization(matrix, types):
    ind_profit = np.where(types == 1)
    ind_cost = np.where(types == -1)
    nmatrix = np.zeros(np.shape(matrix))
    nmatrix[:, ind_profit] = matrix[:, ind_profit] / (np.amax(matrix[:, ind_profit], axis = 0))
    nmatrix[:, ind_cost] = np.amin(matrix[:, ind_cost], axis = 0) / matrix[:, ind_cost]
    return nmatrix


# min-max normalization
def minmax_normalization(X, criteria_type):
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    ind_profit = np.where(criteria_type == 1)
    ind_cost = np.where(criteria_type == -1)

    x_norm[:, ind_profit] = (X[:, ind_profit] - np.amin(X[:, ind_profit], axis = 0)
                             ) / (np.amax(X[:, ind_profit], axis = 0) - np.amin(X[:, ind_profit], axis = 0))

    x_norm[:, ind_cost] = (np.amax(X[:, ind_cost], axis = 0) - X[:, ind_cost]
                           ) / (np.amax(X[:, ind_cost], axis = 0) - np.amin(X[:, ind_cost], axis = 0))

    return x_norm


# max normalization
def max_normalization(X, criteria_type):
    maximes = np.amax(X, axis=0)
    ind = np.where(criteria_type == -1)
    X = X/maximes
    X[:,ind] = 1-X[:,ind]
    return X


# sum normalization
def sum_normalization(X, criteria_type):
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    ind_profit = np.where(criteria_type == 1)
    ind_cost = np.where(criteria_type == -1)

    x_norm[:, ind_profit] = X[:, ind_profit] / np.sum(X[:, ind_profit], axis = 0)

    x_norm[:, ind_cost] = (1 / X[:, ind_cost]) / np.sum((1 / X[:, ind_cost]), axis = 0)

    return x_norm


# equal weighting
def mean_weighting(X):
    N = np.shape(X)[1]
    return np.ones(N) / N


# entropy weighting
def entropy_weighting(X):
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    pij = sum_normalization(X, criteria_type)

    Ej = np.zeros(np.shape(pij)[1])
    for el in range(0, np.shape(pij)[1]):
        if np.any(pij[:, el] == 0):
            Ej[el] = 0
        else:
            Ej[el] = - np.sum(pij[:, el] * np.log(pij[:, el]))

    Ej = Ej / np.log(X.shape[0])

    wagi = (1 - Ej) / (np.sum(1 - Ej))
    return wagi


# standard deviation weighting
def std_weighting(X):
    stdv = np.std(X, axis = 0)
    return stdv / np.sum(stdv)


# CRITIC weighting
def critic_weighting(X):
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    x_norm = minmax_normalization(X, criteria_type)
    std = np.std(x_norm, axis = 0)
    n = np.shape(x_norm)[1]
    correlations = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            correlations[i, j], _ = pearsonr(x_norm[:, i], x_norm[:, j])

    difference = 1 - correlations
    suma = np.sum(difference, axis = 0)
    C = std * suma
    w = C / (np.sum(C, axis = 0))
    return w


# weighted spearman coefficient rw
def weighted_spearman(R, Q):
    N = len(R)
    denom = N**4 + N**3 - N**2 - N
    reszta = (N-R+1)+(N-Q+1)
    suma = 6*sum((R-Q)**2*reszta)
    rW = 1-(suma/denom)
    return rW

# rank similarity coefficient WS
def coeff_WS(R, Q):
    sWS = 0
    N = len(R)
    for i in range(N):
        sWS += 2**(-int(R[i]))*(abs(R[i]-Q[i])/max(abs(R[i] - 1), abs(R[i] - N)))
    WS = 1 - sWS
    return WS
