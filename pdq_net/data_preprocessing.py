import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import config

def load_data(file_path):
    lengths = []
    with open(file_path, 'r') as file:
        for line in file:
            length_str = line.strip().split(';')[1]
            length_list = list(map(int, length_str.split('\t')))
            lengths.append(length_list)
    return np.array(lengths, dtype=object)

def normalize_lengths(lengths):
    max_length = max([max(length) for length in lengths])
    normalized_lengths = [np.array(length) / max_length for length in lengths]
    return normalized_lengths

def pad_sequences(sequences, max_len):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            seq = np.pad(seq, (0, max_len - len(seq)), 'constant')
        padded_sequences.append(seq)
    return padded_sequences

def convert_to_spike_trains(lengths, num_steps=100, max_len=None):
    if max_len is None:
        max_len = max([len(seq) for seq in lengths])
    lengths = pad_sequences(lengths, max_len)
    spike_trains = []
    for i in range(len(lengths)):
        spike_train = torch.zeros(num_steps, max_len)
        for j in range(len(lengths[i])):
            spike_intensity = int(lengths[i][j] * num_steps)
            if spike_intensity < num_steps:
                spike_train[spike_intensity, j] = 1
        spike_trains.append(spike_train)
    return torch.stack(spike_trains)

def save_preprocessed_data(spike_trains, labels, output_dir, split_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'{split_name}_data.pt')
    torch.save({'spike_trains': spike_trains, 'labels': labels}, output_file)

def preprocess_data():
    data_dir = config.DATA_DIR
    output_dir = config.PROCESSED_DATA_DIR
    num_steps = config.NUM_STEPS

    label_map = {}
    all_lengths = []
    all_labels = []
    current_label = 0

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.num'):
            base_name = file_name.split('.')[0]
            if base_name not in label_map:
                label_map[base_name] = current_label
                current_label += 1
            class_label = label_map[base_name]
            file_path = os.path.join(data_dir, file_name)

            lengths = load_data(file_path)
            normalized_lengths = normalize_lengths(lengths)
            all_lengths.extend(normalized_lengths)
            all_labels.extend([class_label] * len(normalized_lengths))

    if all_lengths and all_labels:
        max_len = max([len(seq) for seq in all_lengths])
        spike_trains = convert_to_spike_trains(all_lengths, num_steps=num_steps, max_len=max_len)
        labels = torch.tensor(all_labels)

        print(f"Actual input dimension (max_len): {max_len}")  # 打印实际输入维度

        X_train, X_test, y_train, y_test = train_test_split(spike_trains, labels, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

        save_preprocessed_data(X_train, y_train, output_dir, 'train')
        save_preprocessed_data(X_test, y_test, output_dir, 'test')

        print(f"Preprocessed train data saved to {output_dir}/train_data.pt")
        print(f"Preprocessed test data saved to {output_dir}/test_data.pt")
    else:
        print("No valid data found for processing.")

if __name__ == "__main__":
    preprocess_data()
