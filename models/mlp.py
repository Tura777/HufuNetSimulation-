# reference:  the architecture of the model is directly taken from https://github.com/Bellafqira/DICTION

import torch
import torch.nn as nn


class MLP(nn.Module):
    """work well on mnist ! acc=98.3 with 60 epochs, lr=0.001, opt=Adam, bs = 512"""
    "Ref : https://share.cocalc.com/share/32b94ee413d02759d719862907bb0a85a76c43f1/2016-11-07-175929.pdf "
    def __init__(self, width=28, high=28):
        super().__init__()
        self.fc1 = nn.Linear(width * high, 512)
        self.fc2 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
        self.bt_n = nn.BatchNorm1d(512)

    def forward(self, x):
        out = torch.flatten(x, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.bt_n(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

if __name__ == "__main__":
    mlp = MLP()

    print("\nModel Architecture:\n")
    print(mlp)

    print("\nState Dictionary Keys:\n")
    for key in mlp.state_dict().keys():
        print(key)

    print("\nState Dictionary:\n")
    for key, value in mlp.state_dict().items():
        print(f"{key}: {value.shape}")
