from torch.utils.data import DataLoader
from torch import nn, optim
from imagenet22k_dataset import IN22KDATASET
# Initialize the dataset
# Define your model, loss function and optimizer
# model = YourModel()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# Perform k-fold cross-validation
def test_IN22KDATASET():
    
    root_dir = 'D:\Learning\Grad_0\Project\Swin-Transformer_Tumor\Swin-Transformer_Tumor\data\dataset'
    k_folds = 5
    current_fold = 0

    # Create an instance of IN22KDATASET
    dataset = IN22KDATASET(root_dir, k_folds, current_fold)

    # Get the total number of data
    print(f'Total number of data: {len(dataset)}')

    data, label = dataset[0]
    print(f'First data: {data}, label: {label}')

    data, label = dataset[1848]
    print(f'Last data: {data}, label: {label}')

if __name__ == '__main__':
    test_IN22KDATASET()
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