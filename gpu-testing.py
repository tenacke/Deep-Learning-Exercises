import sys

sys.path.append("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
