from torch.utils.data import DataLoader

if __name__ == "__main__":
    from preprocess_data import load_data
    from neural_network import SimpleCNN
    import matplotlib.pyplot as plt
    import torch

    LEARNING_RATE = 0.05
    MINI_BATCH_SIZE = 64
    SOFTMAX = False

    model = SimpleCNN(softmax=SOFTMAX)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    def train_loop(model, loss_fn, optimiser):
   
        losses = []

        model.train()

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
                    print(f"Training Batch {training_batch + 1}, Mini Batch {batch}, Loss: {loss.item()}")

        return losses
    
    test_data, test_labels_one_hot = load_data("test")
    test_dataloader = DataLoader(list(zip(test_data, test_labels_one_hot)), batch_size=MINI_BATCH_SIZE)

    def testing_loop(model, loss_fn, test_dataloader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch, (images, labels_one_hot) in enumerate(test_dataloader):
                pred = model(images)
                test_loss += loss_fn(pred, labels_one_hot).item()
                correct += (pred.argmax(1) == labels_one_hot.argmax(1)).type(torch.float).sum().item()

        test_loss /= len(test_dataloader.dataset)
        accuracy = correct / len(test_dataloader.dataset)
        print(f"Test Loss: {test_loss}, Accuracy: {accuracy}")
    
    train_losses = train_loop(model, loss_fn, optimiser)
    testing_loop(model, loss_fn, test_dataloader)
    
    plt.plot(train_losses)
    plt.show()