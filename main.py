from torch.utils.data import DataLoader

if __name__ == "__main__":
    from preprocess_data import load_data
    from neural_network import SimpleCNN
    import matplotlib.pyplot as plt
    import torch

    LEARNING_RATE = 0.1
    MINI_BATCH_SIZE = 64
    SOFTMAX = False
    NUMBER_OF_EPOCHS = 5

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
        number_of_batches = len(test_dataloader)
        number_of_samples = len(test_dataloader.dataset)
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels_one_hot in test_dataloader:
                pred = model(images)
                test_loss += loss_fn(pred, labels_one_hot).item()
                correct += (pred.argmax(1) == labels_one_hot.argmax(1)).type(torch.float).sum().item()

        test_loss /= number_of_batches
        accuracy = correct / number_of_samples
        print(f"Test Loss: {test_loss}, Accuracy: {accuracy}")
        return test_loss, accuracy
    
    train_losses = []
    test_losses = []

    for epoch in range(NUMBER_OF_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        train_epoch_losses = train_loop(model, loss_fn, optimiser)
        train_losses.append(train_epoch_losses)
        test_loss, accuracy = testing_loop(model, loss_fn, test_dataloader)
        test_losses.append(test_loss)

    print("Done!")

    # Initialize a list to store the average training loss for each epoch for plotting the training progress over time
    y = []

    # Calculate the average training loss for each epoch and store it in the list y for plotting
    for train_loss in train_losses:
        sum_of_losses = sum(train_loss)
        average_loss = sum_of_losses / len(train_loss)
        y.append(average_loss) 

    # Create a list of epoch numbers for plotting the training and testing losses over time
    x = [i + 1 for i in range(len(y))]  

    # Plot the training and testing losses over epochs to visualize the training process and evaluate the model's performance over time
    plt.plot(x, y)  # Plot the training losses over epochs to visualize the training process
    plt.plot(x, test_losses)  # Plot the test losses over epochs to visualize the testing process
    plt.xlabel("Epoch")  # Label for the x-axis of the plot
    plt.ylabel("Average Loss")  # Label for the y-axis of the plot
    plt.title("Training and Testing Loss")  # Title for the plot
    plt.legend(["Training Loss", "Testing Loss"])  # Add a legend to the plot to differentiate between training and testing losses
    plt.show()