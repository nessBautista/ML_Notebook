# %%
import sklearn
from MLBookCamp.DataRegister import DataRegister
import numpy as np
import matplotlib.pyplot as plt
import random

# Get raw data
datasets_path = "../../datasets"
dr = DataRegister(datasets_path)
dr.get_csv_keys()
raw_data = np.loadtxt(dr.get_path_from_key("linear01"), delimiter=',')

# Plot raw data
raw_x = raw_data[:,:-1]
raw_y = raw_data[:,-1]
raw_y = raw_y[..., np.newaxis]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(raw_x, raw_y)



# MSE
def get_mse(y_pred, y_val):
    mse = ((y_pred-y_val)**2).mean()
    return mse

def predict(theta, X):
    y_pred = []
    for row in X:        
        pred = theta.T.dot(row)
        y_pred.append(pred)
    return np.array(y_pred)

def gradient_descent(X, y, theta):
    eta = 0.1
    n_iterations = 1000
    m = len(X)
    
    for iteration in range(n_iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta = theta - eta * gradients
    return theta

def normal_eq(X, y, theta):
    theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(raw_y)
    return theta_best
    
# Create a random line and plot it
theta = np.random.randn(2,1)
X = np.c_[np.ones((100,1)), raw_x]
print(f'random values of theta:{theta}')
y_pred = predict(theta, X)
ax.plot(raw_x, y_pred)

# Calculate with normal equation
norm_eq_theta = normal_eq(X,raw_y, theta)
print(f'theta using the normal equation:{norm_eq_theta}')
y_pred = predict(norm_eq_theta, X)
ax.plot(raw_x, y_pred, c='orange')

# Calculate with Gradient Descent
gd_theta = gradient_descent(X, raw_y, theta)
print(f'theta after gradient descent:{gd_theta}')
ax.plot(raw_x, y_pred, c='purple')

# Calculate with scikit learn Linear Regresssion
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X, raw_y)
print(linear_regression.intercept_,linear_regression.coef_)

# # Try Ridge regression
from sklearn.linear_model import Ridge

def analyze_ridge_regression(X, y,alphas):    
    predictions=[]
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver='cholesky')
        model.fit(X, y)
        predictions.append(model.predict(X))
    preds = np.array(predictions)
    #PLOT
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(raw_x, raw_y)

    for (pred, alpha) in zip(predictions, alphas):
        ax.plot(raw_x, pred, label=f'alpha:{alpha}')
        ax.legend()

analyze_ridge_regression(X, raw_y, [0, 100, 1000])

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

def analyze_ridge_poly(X, y,alphas):
    poly_features = PolynomialFeatures(degree=10, include_bias=False)
    X_poly = poly_features.fit_transform(X, y)

    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly)

    predictions=[]
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver='cholesky')
        model.fit(X_poly_scaled, y)
        predictions.append(model.predict(X_poly_scaled))
    
    #PLOT
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(raw_x, raw_y)
    
    for (pred, alpha) in zip(predictions, alphas):
        ax.scatter(X, pred, label=f'alpha:{alpha}')
        ax.legend()



analyze_ridge_poly(raw_x,raw_y,[0, 0.0000001, 1, 10])



