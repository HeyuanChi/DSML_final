import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


def inference(model, 
              data, 
              input_length,
              output_length,
              output_steps=10):

    model.eval()
    
    data_t = torch.FloatTensor(data).to(device)
    assert data_t.size(0) >= input_length, "The length of data should at least be input_length"
        
    with torch.no_grad():
        for step in range(output_steps):
            src = data_t[-input_length:].unsqueeze(0)
            tgt = torch.zeros((1, output_length, src.size(-1)), dtype=torch.float).to(device)
            tgt[:, 0, :] = src[:, -1, :]
            
            pred = model(src, tgt).squeeze(0)
            data_t = torch.cat([data_t, pred], dim=0)
            
    data_out = data_t.cpu().numpy()
    return data_out
