from keras.datasets import mnist
from keras import models, layers
from tensorflow.keras.utils import to_categorical


# importing data (in this case it is mnist dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# create network
network = models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation="softmax"))

# Compiling of ANN
network.compile(optimizer="rmsprop",
                loss="categorical_crossentropy",
                metrics=["accuracy"])

# Preparing of inputs and labels
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# Training of ANN
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Testing and Evaluation part
prediction = network.predict(test_images)
test_loss, test_acc = network.evaluate(test_images, test_labels)
