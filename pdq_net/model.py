import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = nn.ConstantPad1d((padding - dilation, 0), 0)  # 确保因果性
        self.bn1 = nn.BatchNorm1d(n_outputs)  # 添加BatchNorm层
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = nn.ConstantPad1d((padding - dilation, 0), 0)  # 确保因果性
        self.bn2 = nn.BatchNorm1d(n_outputs)  # 添加BatchNorm层
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        # 确保残差连接尺寸匹配
        if out.size(2) != res.size(2):
            if res.size(2) > out.size(2):
                res = res[:, :, :out.size(2)]  # 裁剪输入以匹配输出尺寸
            else:
                pad_size = out.size(2) - res.size(2)
                res = F.pad(res, (0, pad_size))  # 填充输入以匹配输出尺寸

        return self.relu(out + res)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.q_linear = nn.Linear(d_model, n_heads * d_k)
        self.k_linear = nn.Linear(d_model, n_heads * d_k)
        self.v_linear = nn.Linear(d_model, n_heads * d_v)
        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        seq_len = q.size(1)

        # 线性变换并调整形状
        q = self.q_linear(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)

        # 计算注意力输出
        attn_output, _ = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_v)
        output = self.dropout(self.fc(attn_output))

        # 更新残差连接的计算，确保形状一致
        residual = q.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_k)
        if residual.size(2) != output.size(2):
            residual = self.fc(residual)  # 使用全连接层匹配尺寸

        output = self.layer_norm(output + residual)

        return output


class FourWayTCNWithAttention(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, n_heads, d_k, d_v, num_steps, input_dim):
        super(FourWayTCNWithAttention, self).__init__()
        self.forward_tcn = self._build_tcn(num_inputs, num_channels, kernel_size, dropout)
        self.backward_tcn = self._build_tcn(num_inputs, num_channels, kernel_size, dropout)
        self.middle_outward_tcn = self._build_tcn(num_inputs, num_channels, kernel_size, dropout)
        self.middle_inward_tcn = self._build_tcn(num_inputs, num_channels, kernel_size, dropout)
        self.attention_fw = AttentionLayer(d_model=config.D_MODEL, n_heads=n_heads, d_k=d_k, d_v=d_v, dropout=dropout)
        self.attention_bw = AttentionLayer(d_model=config.D_MODEL, n_heads=n_heads, d_k=d_k, d_v=d_v, dropout=dropout)
        self.attention_mo = AttentionLayer(d_model=config.D_MODEL, n_heads=n_heads, d_k=d_k, d_v=d_v, dropout=dropout)
        self.attention_mi = AttentionLayer(d_model=config.D_MODEL, n_heads=n_heads, d_k=d_k, d_v=d_v, dropout=dropout)

    def _build_tcn(self, num_inputs, num_channels, kernel_size, dropout):
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        return nn.Sequential(*layers)

    def forward(self, x):
        # 前向TCN
        forward_out = self.forward_tcn(x)

        # 后向TCN
        backward_out = self.backward_tcn(torch.flip(x, [2]))
        backward_out = torch.flip(backward_out, [2])

        # 中间向外TCN
        middle_outward_out = self.middle_outward_tcn(x)

        # 中间向内TCN
        middle_inward_out = self.middle_inward_tcn(x)

        # 注意力机制
        forward_out = forward_out.permute(0, 2, 1)  # (batch_size, seq_len, features)
        backward_out = backward_out.permute(0, 2, 1)
        middle_outward_out = middle_outward_out.permute(0, 2, 1)
        middle_inward_out = middle_inward_out.permute(0, 2, 1)

        forward_attn = self.attention_fw(forward_out, forward_out, forward_out)
        backward_attn = self.attention_bw(backward_out, backward_out, backward_out)
        middle_outward_attn = self.attention_mo(middle_outward_out, middle_outward_out, middle_outward_out)
        middle_inward_attn = self.attention_mi(middle_inward_out, middle_inward_out, middle_inward_out)

        # 合并输出
        combined_out = torch.cat((forward_attn, backward_attn, middle_outward_attn, middle_inward_attn), dim=2)

        combined_out = combined_out.contiguous().view(combined_out.size(0), -1)  # 为全连接层展平

        return combined_out


class FourWayTCNWithAttentionFC(FourWayTCNWithAttention):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, n_heads, d_k, d_v, num_classes, num_steps, input_dim):
        super().__init__(num_inputs, num_channels, kernel_size, dropout, n_heads, d_k, d_v, num_steps, input_dim)
        self.fc = None

    def forward(self, x):
        output = super().forward(x)
        if self.fc is None:
            self.fc = nn.Linear(output.size(1), config.FC_OUT_FEATURES)  # 动态设置全连接层的输入维度
        output = self.fc(output)
        return output
