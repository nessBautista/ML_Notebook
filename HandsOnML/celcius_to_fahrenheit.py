import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#Setup training data
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
    print("{} degrees in Celcius = {} degrees in Fahrenheit".format(c,fahrenheit_a[i]))

#Create the model
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

#Assemble layers into the model
model = tf.keras.Sequential([l0])

#Compile the model, with loss and optimizer functions
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

#Train the model
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

#Display training statistics
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

#Use the model to predict values
print(model.predict([100.0]))

#Looking at the layer weights
print("These are the layer variables: {}".format(l0.get_weights()))

#A little experiment:
#Just for fun, what if we created more Dense layers with different units, which therefore also has more variables?
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
print(model.predict([100.0]))
print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(model.predict([100.0])))
print("These are the l0 variables: {}".format(l0.get_weights()))
print("These are the l1 variables: {}".format(l1.get_weights()))
print("These are the l2 variables: {}".format(l2.get_weights()))