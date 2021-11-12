import numpy as np
import tensorflow as tf
import matplotlib
X = tf.constant(range(10), dtype=tf.float32)
Y = 2 * X + 10
print(X)
print(Y)

#Test data set
X_test = tf.constant(range(10,20), dtype=tf.float32)
Y_test = 2 * X_test + 10

print(X_test)
print(Y_test)

#Get the mean
y_mean = Y.numpy().mean()

def predict_mean(X):
    y_hat = [y_mean]*len(X)
    return y_hat

Y_hat = predict_mean(X_test)
print(Y_hat)


# calculate the loss
errors = (Y_hat -Y)**2
loss = tf.reduce_mean(errors)
print(errors)
print(loss.numpy())

