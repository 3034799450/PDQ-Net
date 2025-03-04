import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # 导入 tqdm 库
import config
from model import FourWayTCNWithAttentionFC

def train_model(train_loader, model, criterion, optimizer, num_epochs, model_save_path):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")  # 创建进度条
        for i, (spike_trains, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            outputs = model(spike_trains)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))  # 更新进度条信息

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # 保存整个模型对象
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    torch.save(model, model_save_path)

if __name__ == "__main__":
    train_data = torch.load(os.path.join(config.PROCESSED_DATA_DIR, 'train_data.pt'))
    spike_trains = train_data['spike_trains']
    labels = train_data['labels']
    train_dataset = TensorDataset(spike_trains, labels)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    input_dim = spike_trains.size(2)

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
        input_dim=input_dim  # 动态传递 input_dim
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model_save_path = os.path.join(config.MODEL_DIR, 'spike_tcn_attention.pth')

    train_model(train_loader, model, criterion, optimizer, num_epochs=config.NUM_EPOCHS, model_save_path=model_save_path)
