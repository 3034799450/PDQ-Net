import os
import time
import torch
from memory_profiler import memory_usage
from data_generator import get_data_loader
from model import FourWayTCNWithAttentionFC
import config

def evaluate_efficiency(model, data_loader):
    # 记录推理时间
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in data_loader:
            _ = model(inputs)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.4f} seconds for {len(data_loader)} batches")

    # 使用 memory_profiler 测量内存使用
    def memory_profiled_inference():
        with torch.no_grad():
            for inputs, labels in data_loader:
                _ = model(inputs)

    mem_usage = memory_usage(proc=memory_profiled_inference)
    max_memory = max(mem_usage)
    print(f"Peak Memory Usage: {max_memory:.2f} MiB")

    return inference_time, max_memory

if __name__ == "__main__":
    # 加载训练好的模型
    model_path = os.path.join(config.MODEL_DIR, 'spike_tcn_attention.pth')
    model = torch.load(model_path)
    model.eval()

    # 加载测试数据
    test_data_path = os.path.join(config.PROCESSED_DATA_DIR, 'test_data.pt')
    test_loader = get_data_loader(test_data_path, batch_size=config.BATCH_SIZE, shuffle=False)

    # 评估模型效率
    inference_time, max_memory = evaluate_efficiency(model, test_loader)

    # 打印模型大小
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # 转换为 MB
    print(f"Model Size: {model_size:.2f} MB")
