import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).unsqueeze(1).float()
        self.y = torch.argmax(torch.tensor(y), 1).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].to(device), self.y[idx].to(device)

def get_dataloaders(X, Y, opt):
    n_samples = len(X)
    train_end = int(0.7 * n_samples)  # 前70%作为训练+验证集
    test_start = int(0.8 * n_samples)  # 后20%作为测试集（中间10%不用，防止相邻重叠）

    x_train_val = X[:train_end]
    y_train_val = Y[:train_end]

    x_test = X[test_start:]
    y_test = Y[test_start:]

    train_size = int(0.8 * len(x_train_val))
    x_train = x_train_val[:train_size]
    y_train = y_train_val[:train_size]
    x_valid = x_train_val[train_size:]
    y_valid = y_train_val[train_size:]

    train_dl = DataLoader(MyDataset(x_train, y_train), batch_size=opt.batch_size, shuffle=True)
    valid_dl = DataLoader(MyDataset(x_valid, y_valid), batch_size=opt.batch_size, shuffle=False)
    test_dl = DataLoader(MyDataset(x_test, y_test), batch_size=opt.batch_size, shuffle=False)
    
    return train_dl, valid_dl, test_dl
