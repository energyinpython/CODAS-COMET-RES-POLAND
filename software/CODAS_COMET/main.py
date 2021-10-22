# CODAS-COMET method

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from itertools import product
import matplotlib.pyplot as plt
from codas import *
from topsis import *
from additions import *
import copy
import os
import seaborn as sns
import matplotlib

# procedures for COMET method
def tfn(x, a, m, b):
    if x < a or x > b:
        return 0
    elif a <= x < m:
        return (x-a) / (m-a)
    elif m < x <= b:
        return (b-x) / (b-m)
    elif x == m:
        return 1


def evaluate_alternatives(C, x, ind):
    if ind == 0:
        return tfn(x, C[ind], C[ind], C[ind + 1])
    elif ind == len(C) - 1:
        return tfn(x, C[ind - 1], C[ind], C[ind])
    else:
        return tfn(x, C[ind - 1], C[ind], C[ind + 1])


#create characteristic values
def get_characteristic_values(matrix):
    cv = np.zeros((matrix.shape[1], 3))
    for j in range(matrix.shape[1]):
        cv[j, 0] = np.min(matrix[:, j])
        cv[j, 1] = np.mean(matrix[:, j])
        cv[j, 2] = np.max(matrix[:, j])
    return cv


#comet algorithm
def COMET(matrix, weights, criteria_types, co_evaluation_method, method_distance):
    # generate characteristic values
    cv = get_characteristic_values(matrix)
    df_char_val = pd.DataFrame(cv)
    #df_char_val.to_csv('Char_val_' + str(year) + '.csv')
    #print(cv)
    # generate matrix with COs
    # cartesian product of characteristic values for all criteria
    co = product(*cv)
    co = np.array(list(co))

    # calculate vector SJ using chosen MCDA method
    sj = co_evaluation_method(co, weights, criteria_types, method_distance)

    # calculate vector P
    k = np.unique(sj).shape[0]
    p = np.zeros(sj.shape[0], dtype=float)

    if co_evaluation_method.__name__ == 'CODAS':
        print(co_evaluation_method.__name__)
        for i in range(1, k):
            ind = sj == np.max(sj)
            p[ind] = (k - i) / (k - 1)
            sj[ind] = np.min(sj) - 1

    elif co_evaluation_method.__name__ == 'TOPSIS':
        print(co_evaluation_method.__name__)
        for i in range(1, k):
            ind = sj == np.max(sj)
            p[ind] = (k - i) / (k - 1)
            sj[ind] = 0


    # inference and obtaining preference for alternatives
    preferences = []

    for i in range(len(matrix)):
        alt = matrix[i, :]
        W = []
        score = 0

        for i in range(len(p)):
            for j in range(len(co[i])):
                for index in range(len(cv[j])):
                    if cv[j][index] == co[i][j]:
                        ind = index
                        break
                W.append(evaluate_alternatives(cv[j], alt[j], ind))
            score += np.product(W) * p[i]
            W = []
        preferences.append(score)
    preferences = np.asarray(preferences)

    rankingPrep = np.argsort(-preferences)
    rank = np.argsort(rankingPrep) + 1

    return preferences, rank



# main
# PROGRAM codas topsis comet

path = '../../DATASET'
file = 'dataset_energy_voivodeships_2019.csv'
pathfile = os.path.join(path, file)
dane = pd.read_csv(pathfile)

# choose investigated year: 2015, 2016, 2017, 2018 2019
year = 2019
dataset = dane[dane['Year'] == year]
data = dataset.iloc[:, 2:]
data = data.reset_index()
data = data.drop(['index'], axis = 1)
data = data.set_index('Ai')
print(data)

list_alt_names = []
for i in range(1, len(data) + 1):
    list_alt_names.append(r'$A_{' + str(i) + '}$')

df_writer_pref = pd.DataFrame()
df_writer_pref['Ai'] = list_alt_names

df_writer_rank = pd.DataFrame()
df_writer_rank['Ai'] = list_alt_names


matrix = data.to_numpy()
types = np.ones(np.shape(matrix)[1])

df_writer = pd.DataFrame()
df_writer['Ai'] = list_alt_names

# choose weighting method
weights = critic_weighting(matrix)

# choose distance metrics
list_methods_distance = [euclidean_distance,
                         hausdorff_distance,
                         #correlative_distance,
                         chebyshev_distance,
                         #std_euclidean_distance,
                         #cosine_distance,
                         ]

#0 - CODAS, 1 - TOPSIS
for it in range(0, 2):
    for method_distance in list_methods_distance:

        if it == 0:
            co_evaluation_method = CODAS

        elif it == 1:
            co_evaluation_method = TOPSIS

        pref, rank = COMET(matrix, weights, types, co_evaluation_method, method_distance)

        if it == 0:
            df_writer_pref[r'$COM + COD_{' + method_distance.__name__[:3] + '}$'] = pref
            df_writer_rank[r'$COM + COD_{' + method_distance.__name__[:3] + '}$'] = rank
        elif it == 1:
            df_writer_pref[r'$COM + TOP_{' + method_distance.__name__[:3] + '}$'] = pref
            df_writer_rank[r'$COM + TOP_{' + method_distance.__name__[:3] + '}$'] = rank


df_writer_pref = df_writer_pref.set_index('Ai')
df_writer_rank = df_writer_rank.set_index('Ai')
df_writer_pref.to_csv('results_pref_' + str(year) + '.csv')
df_writer_rank.to_csv('results_rank_' + str(year) + '.csv')

df_writer_all = pd.concat([df_writer_pref, df_writer_rank], axis = 1)
df_writer_all.to_csv('results_all_' + str(year) + '.csv')


#plots

# creation of correlation matrices
method_types = ['$COM + COD_{euc}$', 
                        '$COM + COD_{hau}$', 
                        '$COM + COD_{che}$',
                        '$COM + TOP_{euc}$',
                        '$COM + TOP_{hau}$',
                        '$COM + TOP_{che}$',
                        ]

dict_new_heatmap_rw = {'$COM + COD_{euc}$': [], 
                        '$COM + COD_{hau}$': [], 
                        '$COM + COD_{che}$': [],
                        '$COM + TOP_{euc}$': [],
                        '$COM + TOP_{hau}$': [],
                        '$COM + TOP_{che}$': [],
                        }


dict_new_heatmap_ws = copy.deepcopy(dict_new_heatmap_rw)

for i in method_types[::-1]:
    for j in method_types:
        dict_new_heatmap_rw[j].append(weighted_spearman(df_writer_rank[i], df_writer_rank[j]))
        dict_new_heatmap_ws[j].append(coeff_WS(df_writer_rank[i], df_writer_rank[j]))
        
df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
df_new_heatmap_rw.columns = method_types

df_new_heatmap_ws = pd.DataFrame(dict_new_heatmap_ws, index = method_types[::-1])
df_new_heatmap_ws.columns = method_types

def draw_heatmap(df_new_heatmap, title):
    sns.set(font_scale=1.1)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="PuBuGn",
                          linewidth=1, linecolor='w')
    plt.yticks(va="center")
    plt.title(title)
    plt.tight_layout()


# correlation matrix with rw coefficient
plt.figure(figsize = (7,4))
draw_heatmap(df_new_heatmap_rw, r'$r_w$')
plt.show()

# correlation matrix with WS coefficient
plt.figure(figsize = (7,4))
draw_heatmap(df_new_heatmap_ws, r'$WS$')
plt.show()


# radar plot
matplotlib.rc_file_defaults()

fig=plt.figure()
ax = fig.add_subplot(111, polar=True)

for col in list(df_writer_rank.columns):
    labels=np.array(list(df_writer_rank.index))
    stats = df_writer_rank.loc[labels, col].values

    angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    # close the plot
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    
    list_of_indexes = list(df_writer_rank.index)
    list_of_indexes.append(df_writer_rank.index[0])
    labels=np.array(list_of_indexes)

    ax.plot(angles, stats, 'o-', linewidth=1)
ax.set_thetagrids(angles * 180/np.pi, labels)
ax.grid(True)
ax.set_axisbelow(True)
plt.legend(df_writer_rank.columns, bbox_to_anchor=(1.1, 0.95, 0.3, 0.2), loc='upper left')
plt.tight_layout()
plt.show()
