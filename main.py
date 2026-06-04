import matplotlib.pyplot as plt
import numpy as np

batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch = 2

file = "dataset/" + batches[batch]

data_batch_1 = unpickle(file)

print(data_batch_1.keys())

data = data_batch_1[b'data']
labels = data_batch_1[b'labels']
batch_label = data_batch_1[b'batch_label']
filenames = data_batch_1[b'filenames']
print(data.shape)
print(len(labels))
print(batch_label)
print(len(filenames))

# array = np.arange(27)
# print(array)
# array = array.reshape(3, 3, 3)
# print(array)

example_number = 45

data_example = data[example_number]
data_example = data_example.reshape(3, 32, 32)
label_example = labels[example_number]
file_example = filenames[example_number]
print(label_example)
print(file_example)
print(data_example.shape)

plt.imshow(data_example.transpose(1, 2, 0))
plt.show()