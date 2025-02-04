import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from transformer import TransformerTimeSeriesModel
from dataset import TimeSeriesDataset
from trainer import train_model
from inference import predict

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")



def main():
    parser = argparse.ArgumentParser(description="Transformer for Lorenz Time Series Forecasting")
    parser.add_argument('--dataset', type=str, default='lorenz63', choices=['lorenz63', 'lorenz96'],
                        help='选择使用的数据集:lorenz63 或 lorenz96')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='运行模式:train(训练)或 test(推理)')
    args = parser.parse_args()

    # 超参数设置
    seq_length = 50
    batch_size = 64
    model_dim = 64
    num_heads = 4
    num_layers = 3
    dropout = 0.1
    num_epochs = 20
    learning_rate = 0.001

    # 根据选择的数据集指定文件名
    if args.dataset == 'lorenz63':
        train_file = 'lorenz63_on0.05_train.npy'
        test_file = 'lorenz63_test.npy'
    else:
        train_file = 'lorenz96_on0.05_train.npy'
        test_file = 'lorenz96_test.npy'

    # 加载数据
    train_data = np.load(train_file)
    test_data = np.load(test_file)
    
    # 假设数据形状为 (num_samples, features)
    # 对训练数据进行归一化，使用训练数据的均值和标准差
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    train_data_norm = (train_data - mean) / std
    test_data_norm = (test_data - mean) / std

    # 构建数据集和 DataLoader
    train_dataset = TimeSeriesDataset(train_data_norm, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 判断输入特征维度(若数据为一维，input_dim=1)
    input_dim = train_data.shape[1] if train_data.ndim > 1 else 1

    # 初始化模型
    model = TransformerTimeSeriesModel(input_dim, model_dim, num_heads, num_layers, dropout).to(device)

    if args.mode == 'train':
        print("====== 训练模式 ======")
        model = train_model(model, train_loader, num_epochs, learning_rate)
        # 保存训练好的模型
        torch.save(model.state_dict(), f'transformer_{args.dataset}.pth')
    else:
        print("====== 测试模式 ======")
        # 加载预训练模型
        model.load_state_dict(torch.load(f'transformer_{args.dataset}.pth', map_location=device))

    # 推理:以测试数据的前 seq_length 个时间步作为初始条件，预测剩余时间步
    pred_len = len(test_data_norm) - seq_length
    init_seq = test_data_norm[:seq_length]
    predictions = predict(model, init_seq, pred_len)
    # 将初始序列与预测结果拼接起来，得到完整的预测序列
    full_prediction = np.concatenate((test_data_norm[:seq_length], predictions), axis=0)

    # 可视化:仅绘制第一维(若数据为多维)
    plt.figure(figsize=(12, 6))
    plt.plot(test_data_norm[:, 0], label="真实值")
    plt.plot(full_prediction[:, 0], label="预测值")
    plt.title(f"Transformer 在 {args.dataset} 测试集上的预测结果")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()