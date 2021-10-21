import numpy as np
from itertools import combinations
import pandas as pd
import sys
import os
import math
from codas import CODAS
from topsis import TOPSIS
from additions import *

import seaborn as sns
import matplotlib.pyplot as plt

# plot numeric experiment correlations results
def plot_boxenplot(data, name):
    ax = sns.boxenplot(x = 'variable', y = 'value', hue = 'Method', data = data)
    ax.tick_params(axis='x', rotation = 90)
    ax.set_xlabel('Distance')
    ax.set_ylabel(r'$'+ name + '$')
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center',
    ncol=2, borderaxespad=0., edgecolor = 'black')
    plt.tight_layout()
    plt.savefig('boxenplot_' + name + '.pdf')
    plt.show()


# main
# list of distance metrics tested
list_methods_distance = [euclidean_distance,
                         hausdorff_distance,
                         correlative_distance,
                         chebyshev_distance,
                         std_euclidean_distance,
                         cosine_distance,
                         ]

#list of possible pairs of distance metrics tested
combinations_names = list(combinations(list_methods_distance, 2))


# num of iterations of experiment
iter = 1000
# dictionary for r_w correlations
results_dict_rw = {
    'variable': [],
    'value': [],
    'Method': []
    }

# dictionary for WS correlations
results_dict_WS = {
    'variable': [],
    'value': [],
    'Method': []
    }

path = '../../DATASET'
file = 'dataset_energy_voivodeships_2019.csv'
pathfile = os.path.join(path, file)
dane = pd.read_csv(pathfile)
data = dane.iloc[:, 2:]
data = data.reset_index()
data = data.drop(['index'], axis = 1)
data = data.set_index('Ai')


# determine range of random numbers based on real data to fill decision matrix
rows = len(data.min())
cols = 2
mini = data.min()
maxi = data.max()
random_range = np.zeros((rows, cols))
for i in range(len(mini)):
    random_range[i, 0] = math.floor(mini[i])
    random_range[i, 1] = math.ceil(maxi[i])


# r - num of rows, c - num of cols, r - 16 - num of alts
r, c = 16, len(data.min())

for it in range(iter):
    
    matrix = np.zeros((r, c))
    for i in range(np.shape(random_range)[0]):
        matrix[:, i] = np.random.uniform(random_range[i, 0], random_range[i, 1], size = r)

    # here are only profit criteria
    types = np.ones(matrix.shape[1])
    # determine weights with CRITIC method
    weights = critic_weighting(matrix)
    
    for met_dis in combinations_names:
        #CODAS
        pref1 = CODAS(matrix, weights, types, met_dis[0])
        rankingPrep = np.argsort(-pref1)
        ranking1 = np.argsort(rankingPrep) + 1
    
        pref2 = CODAS(matrix, weights, types, met_dis[1])
        rankingPrep = np.argsort(-pref2)
        ranking2 = np.argsort(rankingPrep) + 1

        coeff_rw = weighted_spearman(ranking1, ranking2)
        results_dict_rw['variable'].append(met_dis[0].__name__[:4] + r' $ \times $ ' + met_dis[1].__name__[:4])
        results_dict_rw['value'].append(coeff_rw)
        results_dict_rw['Method'].append('CODAS')

        coeffws = coeff_WS(ranking1, ranking2)
        results_dict_WS['variable'].append(met_dis[0].__name__[:4] + r' $ \times $ ' + met_dis[1].__name__[:4])
        results_dict_WS['value'].append(coeffws)
        results_dict_WS['Method'].append('CODAS')

        #TOPSIS
        pref1 = TOPSIS(matrix, weights, types, met_dis[0])
        rankingPrep = np.argsort(-pref1)
        ranking1 = np.argsort(rankingPrep) + 1

        pref2 = TOPSIS(matrix, weights, types, met_dis[1])
        rankingPrep = np.argsort(-pref2)
        ranking2 = np.argsort(rankingPrep) + 1

        coeff_rw = weighted_spearman(ranking1, ranking2)
        results_dict_rw['variable'].append(met_dis[0].__name__[:4] + r' $ \times $ ' + met_dis[1].__name__[:4])
        results_dict_rw['value'].append(coeff_rw)
        results_dict_rw['Method'].append('TOPSIS')
        

        coeffws = coeff_WS(ranking1, ranking2)
        results_dict_WS['variable'].append(met_dis[0].__name__[:4] + r' $ \times $ ' + met_dis[1].__name__[:4])
        results_dict_WS['value'].append(coeffws)
        results_dict_WS['Method'].append('TOPSIS')
        

# transform dictionary with rw correlations data to DataFrame
results_pd_rw = pd.DataFrame(results_dict_rw)
# transform dictionary with WS correlations data to DataFrame
results_pd_WS = pd.DataFrame(results_dict_WS)
# plot correlation results
plot_boxenplot(results_pd_rw, 'r_w')
plot_boxenplot(results_pd_WS, 'WS')
