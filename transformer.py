import numpy as np
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_length, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        


# Transformer模型定义（参考 transformer_timeseries.py 实现思路）
class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerTimeSeriesModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(model_dim, input_dim)
    
    def forward(self, src):
        # src shape: (batch, seq_length, input_dim)
        x = self.input_linear(src)  # (batch, seq_length, model_dim)
        # 可选乘以一个缩放因子（也可以用 sqrt(model_dim)）
        x = x * np.sqrt(x.size(-1))
        x = self.pos_encoder(x)
        # Transformer要求输入 shape 为 (seq_length, batch, model_dim)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # 转回 (batch, seq_length, model_dim)
        out = self.output_linear(x)
        return out
