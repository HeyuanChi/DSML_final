import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


def train_model(model, 
                data_loader, 
                num_epochs=10,
                d_model=512,
                warmup_steps=200):
    model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    
    losses = []

    epoch_bar = tqdm(range(1, num_epochs + 1), desc='Epochs', leave=True)

    for step_num in epoch_bar:         
        lrate = min(1 / step_num ** 0.5, step_num / warmup_steps ** 1.5) / d_model ** 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrate
        optimizer.zero_grad()

        x, y = data_loader()            
        x = x.to(device)
        y = y.to(device) 
        
        out = model(x, y[:, :-1, :])
        target = y[:, 1:, :]
            
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        epoch_bar.set_postfix({'Loss': f'{loss.item():.6f}'})

        with open('loss_log.txt', 'a') as f:
            f.write(f"{step_num}, {loss.item():.6f}\n")

        if loss.item() == min(losses):
            torch.save(model.state_dict(), "model.pth")

    return losses
