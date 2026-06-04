import matplotlib.pyplot as plt
import numpy as np

batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

BATCH_NUMBER = 1
EXAMPLE_NUMBER = 23
MAX_PIXEL_VALUE = 255

file = "dataset/" + batches[BATCH_NUMBER - 1]

data_batch_1 = unpickle(file)

print(data_batch_1.keys())

data = data_batch_1[b'data'] / MAX_PIXEL_VALUE
labels = data_batch_1[b'labels']
batch_label = data_batch_1[b'batch_label']
filenames = data_batch_1[b'filenames']
# print(data.shape)
# print(len(labels))
# print(batch_label)
# print(len(filenames))

# array = np.arange(27)
# print(array)
# array = array.reshape(3, 3, 3)
# print(array)


data_example = data[EXAMPLE_NUMBER]
data_example = data_example.reshape(3, 32, 32)
label_example = labels[EXAMPLE_NUMBER]
file_example = filenames[EXAMPLE_NUMBER]
print(data_example.max())

plt.imshow(data_example.transpose(1, 2, 0))
plt.show()