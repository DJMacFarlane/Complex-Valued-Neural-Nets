import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import complexnn
import complexactivations as ca

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# take Fourier transform of the images
x_train_fft = tf.signal.fft2d(tf.cast(x_train, tf.complex64))
x_test_fft = tf.signal.fft2d(tf.cast(x_test, tf.complex64))

# Create the model
model = tf.keras.models.Sequential([
    complexnn.ComplexConv2D(16, (3, 3), activation=ca.cmplx_rrelu, input_shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    complexnn.ComplexDense(64, activation=ca.cmplx_crelu),
    complexnn.ComplexDropout(0.2),
    complexnn.ComplexDense(10, activation=ca.abs_softmax)
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train_fft, y_train, epochs=10, validation_data=(x_test_fft, y_test))

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train_fft, y_train, epochs=10, validation_data=(x_test_fft, y_test))

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Fashion MNIST Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Complex Train', 'Complex Validation', 'Real Train', 'Real Validation'], loc='upper left')
plt.savefig('./figs/fashionmnist.png')

