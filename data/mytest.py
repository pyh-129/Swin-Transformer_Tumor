from torch.utils.data import DataLoader

# 初始化数据集
dataset = IN22KDATASET(root='D:\Learning\Grad_0\Project\Swin-Transformer\data\dataset\shell6', k_folds=5)

# 定义模型、损失函数和优化器
model = YourModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for current_fold in range(dataset.k_folds):
    dataset.current_fold = current_fold

    # 创建训练和验证的数据加载器
    train_loader = DataLoader(Subset(dataset, dataset.train_indices), batch_size=32, shuffle=True)
    valid_loader = DataLoader(Subset(dataset, dataset.test_indices), batch_size=32, shuffle=False)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                # 计算验证指标，如准确率等

        # 打印或保存训练和验证的结果