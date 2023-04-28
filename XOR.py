import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
import complexnn
import complexactivations as ca

# XOR with complex twist
training_data = np.array([[0,0],[0,1],[1,0],[1,1]])
target_data = np.array([[0],[1],[1],[0+1j]])

model = Sequential()

model.add(complexnn.ComplexDense(16, input_dim=2, activation=ca.cmplx_crelu))
model.add(complexnn.ComplexDense(1, activation='tanh'))


model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=1000)
scores = model.evaluate(training_data, target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print (np.round(model.predict(training_data)))


