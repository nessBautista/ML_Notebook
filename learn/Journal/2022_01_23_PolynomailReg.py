# %%
%matplotlib inline
from ast import increment_lineno
import matplotlib
import sklearn
from MLBookCamp.DataRegister import DataRegister
import numpy as np
#np.random.seed(1)
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import Utilities as util
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Build random polynomial
def get_random_polynomial(number_of_samples):
    m = number_of_samples
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

    """ noise = np.random.randn(number_of_samples,1)
    X = 6*np.random.rand(number_of_samples, 1)-3
    y = 0.5 + X**2 + X + 2 + noise """
    return X,y 

X, y = get_random_polynomial(100)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.scatter(X,y)

# Extract poly features from X
polyFeat = PolynomialFeatures(degree=10, include_bias=False)
X_poly = polyFeat.fit_transform(X)

# Now Train a linear model
linear_model = LinearRegression()
linear_model.fit(X_poly, y)

# Test the model and plot it
y_pred = linear_model.predict(X_poly)
X_sorted, y_sorted = util.sort_polynomial(X,y_pred)
ax1.plot(X_sorted,y_sorted, color='orange')

# Plot Learning Curves
def learning_curves(model, X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    m = len(X_train) # Rows
    
    train_errors, val_errors = [], []
    for rows in range(1,m):                
        X_subset = X_train[:rows]
        y_subset = y_train[:rows]
        model.fit(X_subset, y_subset)
        # Test again a subset of traiining data
        y_training_pred = model.predict(X_subset)
        # Validate with all the validation data
        y_val_pred = model.predict(X_val)

        train_error = mean_squared_error(y_train[:rows], y_training_pred)
        val_error = mean_squared_error(y_val, y_val_pred)        
        train_errors.append(train_error)
        val_errors.append(val_error)
    # plot curves
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")    
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    ax.legend()





#learning_curves(LinearRegression(), X_poly, y)
learning_curves(LinearRegression(), X_poly, y)

""" def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
plot_learning_curves(LinearRegression(), X_poly, y) 
 """
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

learning_curves(polynomial_regression, X, y)
