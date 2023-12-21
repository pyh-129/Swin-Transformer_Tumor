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
    
    root_dir = 'D:\Learning\Grad_0\Project\Swin-Transformer_Tumor\Swin-Transformer_Tumor\data\dataset2'
    k_folds = 5
    current_fold = 1

    # Create an instance of IN22KDATASET
    # dataset = IN22KDATASET(root_dir, k_folds, current_fold)D
    # dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config,5,current_fold)
    dataset2 = IN22KDATASET(root_dir,k_folds,current_fold)
    # Get the total number of data
    # print(f'Total number of data: {len(dataset)}')
    # print(len(dataset2))
    # data, label = dataset2[160]
    # print(data.shape)
    # print(f'First data: {data}, label: {label}')
    # print('benig',dataset2.malignant)


    # 打印数据集中的总样本数
    print(f'Total number of data: {len(dataset2)}')
    print(len(dataset2.train_indices))
    print(len(dataset2.test_indices))
    # 遍历数据集中的每个样本，并打印图像和标签
    # for idx in range(len(dataset2)):
    #     data, label = dataset2[idx]
    #     print(f'Sample {idx + 1}: data shape - {data.shape}, label - {label}')
       

    # data, label = dataset[1848]
    # print(f'Last data: {data}, label: {label}')




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