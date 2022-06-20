from keras import layers, models


# Layer is a part of a model
layer1 = layers.Dense(32, input_shape=(784, ))
layer2 = layers.Dense(32)

# Model is a simple unit of Neural Network
model = models.Sequential()
model.add(layer1)
model.add(layer2)

# Overall summary of a model
summ = model.summary()



