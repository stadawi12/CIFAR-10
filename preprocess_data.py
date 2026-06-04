import torch

# Names of the data batches in the dataset
BATCHES = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
MAX_PIXEL_VALUE = 255

# Function to unpickle the data files
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Function to load the data and labels from a specified batch
def load_data(batch_number):
    file_name : str = "dataset/" + BATCHES[batch_number - 1]
    data_batch = unpickle(file_name)
    data = torch.tensor(data_batch[b'data'], dtype=torch.float32)
    data.div_(MAX_PIXEL_VALUE)
    data = data.reshape(-1, 3, 32, 32)
    labels = torch.tensor(data_batch[b'labels'], dtype=torch.long)
    return data, labels

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    BATCH_NUMBER = 4
    EXAMPLE_NUMBER = 23    

    data, labels = load_data(BATCH_NUMBER)

    data_example = data[EXAMPLE_NUMBER] 
    print(data_example.max())

    plt.imshow(data_example.permute(1, 2, 0))
    plt.show()