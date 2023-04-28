# adapted from https://www.tensorflow.org/tutorials/audio/simple_audio
import tensorflow as tf
import CVNN.complexnn as complexnn
import CVNN.complexactivations as ca
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load cats and dogs dataset
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory("emotions/Sub/",
                                                    labels='inferred',
                                                    batch_size=10,
                                                    validation_split=0.2,
                                                    seed=0,
                                                    subset='both')

label_names = np.array(train_ds.class_names)


def squeeze(audio, label):
    return tf.squeeze(audio, axis=-1), label

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# Create Test Set
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)


def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def spectrogram(ds):
    return ds.map(
        map_func = lambda audio, label:(get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

train_spectrogram_ds = spectrogram(train_ds)
val_spectrogram_ds = spectrogram(val_ds)
test_spectrogram_ds = spectrogram(test_ds)

# train_spectrogram_ds = train_spectrogram_ds.take(100).cache().prefetch(tf.data.AUTOTUNE)
# val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
# test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break

input_shape = example_spectrograms.shape[1:]
num_labels = len(label_names)
print(input_shape)

# Create Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Resizing(64, 64),
    complexnn.ComplexConv2D(8, kernel_size=(4, 4), activation=ca.cmplx_crelu),
    complexnn.ComplexDropout(0.3),
    tf.keras.layers.Flatten(),
    complexnn.ComplexDense(32, activation=ca.cmplx_crelu),
    complexnn.ComplexDropout(0.5),
    complexnn.ComplexDense(num_labels, activation=ca.abs_softmax)
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

EPOCHS = 20
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
)

# Plot Training and Validation Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Emotion Classification Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('./figs/emotion_accuracy.png')

y_pred = model.predict(test_spectrogram_ds)
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Emotion Classification Confusion Matrix')
plt.savefig('./figs/confusion_matrix.png')


