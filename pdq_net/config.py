# config.py

DATA_DIR = 'data'  # 原始数据目录
PROCESSED_DATA_DIR = 'processed_data'  # 预处理后数据保存目录
VALUE_DIR = 'value'  # 评估结果保存目录
MODEL_DIR = 'model'  # 模型保存目录


# 在 config.py 文件中添加类别标签列表
CLASS_NAMES = ['Chat', 'Email', 'File Sharing', 'Video', 'VoIP']  # 根据你的实际类别标签修改

NUM_STEPS = 100  # 脉冲序列的时间步数
TEST_SIZE = 0.2  # 测试集所占比例
RANDOM_STATE = 42  # 随机种子
BATCH_SIZE = 32  # 批处理大小
NUM_EPOCHS = 100  # 训练的轮数
LEARNING_RATE = 0.001  # 学习率
NUM_CLASSES = 5  # 分类类别数

FC_OUT_FEATURES = NUM_CLASSES  # 全连接层输出特征数

# TCN和注意力机制的配置项
NUM_CHANNELS = [32, 64, 128]  # TCN的通道数
KERNEL_SIZE = 2  # 卷积核大小
DROPOUT = 0.2  # Dropout率

# 注意力机制的配置项
ATTENTION_HEADS = 8
ATTENTION_D_K = 16  # 调整 d_k 和 d_v 使其与新的 NUM_CHANNELS 匹配
ATTENTION_D_V = 16

# 确保 d_model 的值与 num_channels 和 n_heads * d_k 一致
D_MODEL = NUM_CHANNELS[-1]  # 四向TCN模型的输出维度是NUM_CHANNELS[-1]

# 输入特征维度占位符
INPUT_DIM = 30  # 这个值将在运行时被实际计算的 max_len 覆盖

# 评估模型的选择：'original' 或 'continued'
EVALUATION_MODEL = 'original'
