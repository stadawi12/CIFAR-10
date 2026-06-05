from torch.utils.data import DataLoader

if __name__ == "__main__":
    from preprocess_data import load_data
    from neural_network import SimpleCNN
    import matplotlib.pyplot as plt
    import torch

    LEARNING_RATE = 0.05
    MINI_BATCH_SIZE = 64

    model = SimpleCNN()

    loss_fn = torch.nn.CrossEntropyLoss()

    optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    losses = []

    for (training_batch) in range(5):
        data, labels_one_hot = load_data(training_batch + 1)
        dataset = list(zip(data, labels_one_hot))
        dataloader = DataLoader(dataset, batch_size=MINI_BATCH_SIZE, shuffle=True)

        for batch, (images, labels_one_hot) in enumerate(dataloader):
            pred = model(images)
            loss = loss_fn(pred, labels_one_hot)     
            losses.append(loss.item())    
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            if batch % 10 == 0:
                print(f"Training Batch {training_batch + 1}, Batch {batch}, Loss: {loss.item()}")
            
    plt.plot(losses)
    plt.show()