import torch.nn as nn

class AttackerAE(nn.Module):
    def __init__(self):
        super(AttackerAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 64, kernel_size=4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 96, kernel_size=4),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
def encoder():
    return AttackerAE()
