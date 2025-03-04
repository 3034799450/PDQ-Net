import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_generator import get_data_loader
from model import FourWayTCNWithAttentionFC  # 使用你的模型
import config
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np  # 确保导入 numpy


def plot_confusion_matrix(cm, classes, output_dir, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, f'{title}.png'))
    plt.show()


def plot_confusion_matrix_normalized(cm, classes, output_dir, title='Confusion matrix (Normalized)', cmap=plt.cm.Blues):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.around(cm_normalized * 100, decimals=2)  # 转换为百分比并保留两位小数
    plt.figure(figsize=(10, 5))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, f'{title}.png'))
    plt.show()


def evaluate_model(test_loader, model, output_dir, class_names):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for spike_trains, labels in test_loader:
            outputs = model(spike_trains)  # 仅返回分类结果
            _, predicted = torch.max(outputs, 1)  # 直接使用 outputs，而不是 outputs.data
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Overall Accuracy: {accuracy:.2f}%')

    cm = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(cm, classes=class_names, output_dir=output_dir, title='Confusion matrix (Counts)')
    plot_confusion_matrix_normalized(cm, classes=class_names, output_dir=output_dir,
                                     title='Confusion matrix (Normalized)')

    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    with open(os.path.join(output_dir, 'evaluation.txt'), 'w') as f:
        f.write(f'Overall Accuracy: {accuracy:.2f}%\n')
        f.write("Classification Report:\n")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"{label}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")

    print("Classification Report:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"{label}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

    # 计算每个类别的FPR
    fpr = {}
    for i, class_name in enumerate(class_names):
        fp = cm.sum(axis=0)[i] - cm[i, i]
        tn = cm.sum() - (cm.sum(axis=1)[i] + cm.sum(axis=0)[i] - cm[i, i])
        fpr[class_name] = fp / (fp + tn)

    with open(os.path.join(output_dir, 'evaluation.txt'), 'a') as f:
        f.write("\nFalse Positive Rates:\n")
        for class_name, rate in fpr.items():
            f.write(f"{class_name}: {rate:.4f}\n")

    print("\nFalse Positive Rates:")
    for class_name, rate in fpr.items():
        print(f"{class_name}: {rate:.4f}")


if __name__ == "__main__":
    test_data_path = os.path.join(config.PROCESSED_DATA_DIR, 'test_data.pt')
    test_data = torch.load(test_data_path)
    spike_trains = test_data['spike_trains']
    labels = test_data['labels']
    test_dataset = TensorDataset(spike_trains, labels)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 选择评估的模型
    if config.EVALUATION_MODEL == 'continued':
        model_path = os.path.join('model_continue', 'spike_tcn_attention_continued.pth')
    else:
        model_path = os.path.join(config.MODEL_DIR, 'spike_tcn_attention.pth')

    model = torch.load(model_path)  # 加载整个模型对象

    evaluate_model(test_loader, model, config.VALUE_DIR, class_names=config.CLASS_NAMES)  # 使用类别标签
