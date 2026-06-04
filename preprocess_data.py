import matplotlib.pyplot as plt
import numpy as np
import torch

batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

BATCH_NUMBER = 4
EXAMPLE_NUMBER = 23
MAX_PIXEL_VALUE = 255

file_name : str = "dataset/" + batches[BATCH_NUMBER - 1]

data_batch_1 = unpickle(file_name)

print(data_batch_1.keys())

data = torch.tensor(data_batch_1[b'data'], dtype=torch.float32)
data.div_(MAX_PIXEL_VALUE)
labels = torch.tensor(data_batch_1[b'labels'], dtype=torch.long)
batch_label = data_batch_1[b'batch_label']
filenames = data_batch_1[b'filenames']

data_example = data[EXAMPLE_NUMBER]
data_example = data_example.reshape(3, 32, 32)
label_example = labels[EXAMPLE_NUMBER]
file_example = filenames[EXAMPLE_NUMBER]
print(data_example.max())

plt.imshow(data_example.permute(1, 2, 0))
plt.show()