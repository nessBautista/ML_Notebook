# %%
%matplotlib inline
from ast import increment_lineno
import matplotlib
import sklearn
from MLBookCamp.DataRegister import DataRegister
import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt


datasets_path = "../../datasets"
dr = DataRegister(datasets_path)
raw_data = np.loadtxt(dr.get_path_from_key("linear01"), delimiter=',')

# Polynomial random data
def get_random_polynomial():
    m = 100 # Rows, also: number of instances
    noise = np.random.randn(m, 1)    
    X = 6*np.random.rand(m, 1)-3
    y = 0.5 + X**2 + X + 2 + noise
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.scatter(X,y)
    return np.array(X), np.array(y)

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


X_poly, y_poly = get_random_polynomial()
X, y = sort_polynomial(X_poly, y_poly)
plt.plot(X, y) 


# Enter polynomial
X, y = get_random_polynomial()
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
y_preds = lin_reg.predict(X_poly)
X, y = sort_polynomial(X, y_preds)
print(lin_reg.intercept_, lin_reg.coef_)
plt.plot(X, y, color='orange')

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Learning Curves
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val =  train_test_split(X,y,test_size=0.2)
    train_errors, valid_errors = [],[]
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_pred = model.predict(X_train[:m])
        y_val_pred = model.predict(X_val[:m])
        train_errors.append(mean_squared_error(y_train[:m],y_train_pred))
        valid_errors.append(mean_squared_error(y_val[:m],y_val_pred))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    ax.plot(np.sqrt(valid_errors), "b-+", linewidth=3, label="valid")
    ax.legend()
    return model.predict(X)

X, y = get_random_polynomial()
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
y_pred = plot_learning_curves(lin_reg, X_poly, y)
X, y = sort_polynomial(X_poly, y)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(X,y)
