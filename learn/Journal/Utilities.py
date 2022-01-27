import numpy as np

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_dataset(n_samples=100):
    rnd = np.random.RandomState(42)
    x = np.linspace(-3, 3, n_samples)
    y_no_noise = np.sin(4 * x) + x
    y = y_no_noise + rnd.normal(size=len(x))
    return x, y


def merge(left, right):
    l=0
    r=0    
    sorted = []
    while (l < len(left)) & (r < len(right)):
        if left[l][1]<right[r][1]:
            sorted.append(left[l])
            l +=1
        elif left[l][1]>right[r][1]:
            sorted.append(right[r])
            r += 1
        else:
            sorted.append(left[l])
            sorted.append(right[r])
            l +=1
            r +=1
    
    # add whatever is still in the arrays
    while l < len(left):
        sorted.append(left[l])
        l += 1
    while r < len(right):
        sorted.append(right[r])
        r += 1
    return sorted


def merge_sort_array(x):
    if len(x) < 2:
        return x
    #Split in two
    l,r = np.array_split(x,2)
    left = merge_sort_array(l)
    right = merge_sort_array(r)
    sorted = merge(left, right)
    return np.array(sorted)
    
'''X.shape = (N,1), y.shape = (N,1)
They are regulary a set of scatter points    
'''
def sort_polynomial(X, y):
    # Add indexes in left column
    indexes = np.arange(0, len(X),1)
    X_random = np.c_[indexes, X]

    # Sort X, y     
    sorted_X = merge_sort_array(X_random)
    sorted_idxs = sorted_X[:,0].astype(int)
    sorted_X = sorted_X[:,1]
    sorted_y = y[sorted_idxs]

    return sorted_X, sorted_y

def normalize(array):
    max = np.argmax(array)
    norm = array/max
    return norm

def plot_corr_matrix(data):
    corr = data.corr()
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
);