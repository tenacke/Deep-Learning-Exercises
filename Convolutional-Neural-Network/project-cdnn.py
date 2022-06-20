import numpy as np
import h5py
import matplotlib.pyplot as plt


def zero_pad(X, pad):
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), "constant", constant_values=0)
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z += float(b)
    return Z


def conv_forward(a_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = a_prev.shape
    (f, f, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    n_H = int(((n_H_prev-f+2*pad)/stride)+1)
    n_W = int(((n_H_prev - f + 2 * pad) / stride) + 1)

    Z = np.zeros([m, n_H, n_W, n_C])

    A_prev_pad = zero_pad(a_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f
                    s_slice_prev =  a_prev_pad[vert_start: vert_end, horiz_start: horiz_end]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[...,c], b[...,c])




# Visual settings
plt.rcParams["figure.figsize"] = (5.0, 4.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


# zero padding
np.random.seed(1)
x = np.random.randn(4,3,3,2)
x_pad = zero_pad(x, 2)

fig, axarr = plt.subplots(1,2)
axarr[0].set_title("x")
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title("x_pad")
axarr[1].imshow(x_pad[0,:,:,0])

plt.show()

# convolution with no stride
np.random.seed(1)
a_value = np.random.randn(4, 4, 3)
weight = np.random.randn(4, 4, 3)
b_value = np.random.randn(1, 1, 1)

Z_value = conv_single_step(a_value, weight, b_value)
