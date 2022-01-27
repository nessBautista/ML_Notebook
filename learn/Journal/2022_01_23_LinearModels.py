# %%
from cProfile import label
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
X, y, true_coefficients = make_regression(n_samples=80, n_features=30, n_informative=10, noise=100, coef=True, random_state=5)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=5)
print(X_train.shape)
print(y_train.shape)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# Train a first Linear Regression model
# Training and test
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
pred_train = linear_reg.predict(X_train)
pred_test = linear_reg.predict(X_test)

print(f'train R^2: {linear_reg.score(X_train, y_train)}, from sk:{r2_score(y_train, pred_train)}')
print(f'test R^2: {linear_reg.score(X_test, y_test)}, from sk:{r2_score(y_test, pred_test)}')

# Validation (with the real model, as you know which are the true coefficients)
# This is the best score we could get: The one with the real coefficients
print(r2_score(np.dot(X, true_coefficients), y))

# Visualize the info about the coefficients
coefficient_sorting =np.argsort(true_coefficients)[::-1]
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(true_coefficients[coefficient_sorting],"o", label="True")
ax1.plot(linear_reg.coef_[coefficient_sorting],"o", label="Linear Regression")
ax1.legend()


def plot_ridge_coef_scores():
    ridge_models = {}
    training_scores = []
    test_scores = []
    for alpha in [100,10,1,0.01]:
        ridge = Ridge(alpha=alpha).fit(X_train, y_train)
        training_scores.append(ridge.score(X_train, y_train))
        test_scores.append(ridge.score(X_test, y_test))
        ridge_models[alpha] = ridge
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(training_scores, label="Training scores")
    ax.plot(test_scores, label="Test scores")
    plt.xticks(range(4), [100,10,1,0.01])
    ax.legend(loc="best")
    return ridge_models

ridge_models= plot_ridge_coef_scores()

def plot_ridge_coef_mse():
    ridge_models = {}
    training_mse = []
    test_mse = []
    for alpha in [100,10,1,0.01]:
        ridge = Ridge(alpha=alpha).fit(X_train, y_train)
        training_pred = ridge.predict(X_train)
        test_pred = ridge.predict(X_test)
        training_mse.append(mean_squared_error(y_train,training_pred))
        test_mse.append(mean_squared_error(y_test,test_pred))
        ridge_models[alpha] = ridge
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(training_mse, label="Training mse")
    ax.plot(test_mse, label="Test mse")
    plt.xticks(range(4), [100,10,1,0.01])
    ax.legend(loc="best")
    return ridge_models

ridge_models = plot_ridge_coef_mse()

def plot_coefficients(true_coefficients, models):
    coeff_sorting= np.argsort(true_coefficients)[::-1]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i,alpha in enumerate([100,10,1,0.01]):
        ax.plot(true_coefficients[coefficient_sorting],"o", label="True", c="blue")
        ax.plot(models[alpha].coef_[coeff_sorting], "o", label="alpha=%.2f"%alpha, c=plt.cm.summer(i/3.))
    ax.legend(loc="best")

plot_coefficients(true_coefficients, ridge_models)
