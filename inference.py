import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


# 推理函数：给定初始序列，递归预测未来的 pred_len 个时间步
def predict(model, init_seq, pred_len):
    # init_seq: numpy array, shape (seq_length, input_dim)
    model.eval()
    predictions = []
    # 增加 batch 维度
    input_seq = torch.tensor(init_seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_length, input_dim)
    with torch.no_grad():
        for _ in range(pred_len):
            output = model(input_seq)  # 输出 shape: (1, seq_length, input_dim)
            next_val = output[0, -1, :]  # 取最后一个时间步的预测作为下一个预测
            predictions.append(next_val.cpu().numpy())
            # 将序列向前滑动：删除第一个时间步，追加最新预测
            next_val = next_val.unsqueeze(0).unsqueeze(0)  # (1,1,input_dim)
            input_seq = torch.cat((input_seq[:, 1:, :], next_val), dim=1)
    return np.array(predictions)