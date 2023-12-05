from torch.utils.data import DataLoader
from torch import nn, optim
from imagenet22k_dataset import IN22KDATASET
# Initialize the dataset
dataset = IN22KDATASET(root='D:\Learning\Grad_0\Project\Swin-Transformer\data\dataset\shell6', k_folds=5)

# Define your model, loss function and optimizer
# model = YourModel()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# Perform k-fold cross-validation
for fold in range(dataset.k_folds):
    # Get the data for the current fold
    train_data, test_data = dataset[fold]

    # Create DataLoaders for the training and test data
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    print(train_loader)
    # Train the model for the current fold
    # for epoch in range(num_epochs):
    #     for inputs, targets in train_loader:
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()

    # Evaluate the model on the test data
    # model.eval()
    # with torch.no_grad():
    #     for inputs, targets in test_loader:
    #         outputs = model(inputs)
    #         # Compute your evaluation metrics here