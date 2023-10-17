import tensorflow as tf
import numpy as np
import time

start_time = time.time()

# Check if GPU is available
if tf.config.experimental.list_physical_devices('GPU'):
    print("GPU is available.")
else:
    print("GPU is NOT available.")

print("--- %s seconds to check GPU availability ---" % (time.time() - start_time))

# Load MNIST dataset
x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')
x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')

x_train, x_test = x_train / 255.0, x_test / 255.0

print("--- %s seconds to load data ---" % (time.time() - start_time))

# Build Model
start_time = time.time()
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("--- %s seconds to build and compile model ---" % (time.time() - start_time))

# Train Model
start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=128)
print("--- %s seconds to train model ---" % (time.time() - start_time))

# Evaluate Model
start_time = time.time()
model.evaluate(x_test, y_test)
print("--- %s seconds to evaluate model ---" % (time.time() - start_time))
