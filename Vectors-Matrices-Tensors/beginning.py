import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt

scalar = np.array(7)  # zero dimensional array => scalar
vector = np.array([7, 14, 21])  # one dimensional array => vector
matrix = np.array([[7, 14, 21],
                   [7, 14, 21],
                   [7, 14, 21]])  # two dimensional array => matrix
tensor = np.array([[[7, 14, 21], [7, 14, 21], [7, 14, 21]],
                   [[7, 14, 21], [7, 14, 21], [7, 14, 21]],
                   [[7, 14, 21], [7, 14, 21], [7, 14, 21]]])  # multi dimensional array => matrix
print(np.ndim(scalar), np.ndim(vector), np.ndim(matrix), np.ndim(tensor))

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

x = 0
while x < 5:
    img = train_images[x]
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()
    x += 1
