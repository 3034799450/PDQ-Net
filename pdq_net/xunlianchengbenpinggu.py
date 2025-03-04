import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import config
from model import FourWayTCNWithAttentionFC


def train_model(train_loader, model, criterion, optimizer, num_epochs, model_save_path):
    total_training_start_time = time.time()  # 记录总训练开始时间

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # 记录每轮开始时间
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for i, (spike_trains, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            outputs = model(spike_trains)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        epoch_end_time = time.time()  # 记录每轮结束时间
        epoch_duration = epoch_end_time - epoch_start_time
        print(
            f'Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_duration:.2f} seconds, Loss: {running_loss / len(train_loader):.4f}')

    total_training_end_time = time.time()  # 记录总训练结束时间
    total_training_duration = total_training_end_time - total_training_start_time
    print(f'Total Training Time: {total_training_duration:.2f} seconds')

    # 保存模型
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    torch.save(model, model_save_path)


if __name__ == "__main__":
    # 加载训练数据
    train_data = torch.load(os.path.join(config.PROCESSED_DATA_DIR, 'train_data.pt'))
    spike_trains = train_data['spike_trains']
    labels = train_data['labels']
    train_dataset = TensorDataset(spike_trains, labels)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    input_dim = spike_trains.size(2)

    # 初始化模型
    model = FourWayTCNWithAttentionFC(
        num_inputs=config.NUM_STEPS,
        num_channels=config.NUM_CHANNELS,
        kernel_size=config.KERNEL_SIZE,
        dropout=config.DROPOUT,
        n_heads=config.ATTENTION_HEADS,
        d_k=config.ATTENTION_D_K,
        d_v=config.ATTENTION_D_V,
        num_classes=config.NUM_CLASSES,
        num_steps=config.NUM_STEPS,
        input_dim=input_dim
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 训练模型并记录时间
    model_save_path = os.path.join(config.MODEL_DIR, 'spike_tcn_attention.pth')
    train_model(train_loader, model, criterion, optimizer, num_epochs=config.NUM_EPOCHS,
                model_save_path=model_save_path)
