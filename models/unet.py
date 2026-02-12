import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.ReLU()
        )

        self.up = nn.ConvTranspose2d(128,64,2,stride=2)

        self.dec = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,3,3,padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        up = self.up(e2)
        concat = torch.cat([up, e1], dim=1)

        out = self.dec(concat)
        return out