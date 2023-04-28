import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import complexnn
import complexactivations as ca
from matplotlib import pyplot as plt


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("complex") / 255 
x_test = x_test.astype("complex") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        complexnn.ComplexConv2D(16, kernel_size=(4, 4), activation=ca.cmplx_crelu),
        layers.Flatten(),
        complexnn.ComplexDropout(0.4),
        complexnn.ComplexDense(num_classes, activation=ca.abs_softmax)
    ]
)

model.summary()

batch_size = 256
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')



model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(4, 4), activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ]
)

model.summary()

batch_size = 256
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('MNIST Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Complex Train', 'Complex Test' ,'Real Train', 'Real Test'], loc='upper left')


plt.savefig('./figs/combined.png')

