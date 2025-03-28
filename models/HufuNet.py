import torch.nn as nn

# modified architecture


#
# class HufuNet(nn.Module):
#     def __init__(self):
#         super(HufuNet, self).__init__()
#         # Encoder: compress 28x28 images to a latent representation of shape [64, 4, 4]
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> [16, 14, 14]
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> [32, 7, 7]
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> [64, 4, 4]
#             nn.ReLU()
#         )
#
#         # Decoder: reconstruct the image from the latent representation
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),  # -> [32, 7, 7]
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [16, 14, 14]
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [1, 28, 28]
#             nn.Sigmoid()  # output pixel values in [0, 1]
#         )
#
#     def forward(self, x):
#
#
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded
#
#
# def encoder():
#     return HufuNet()


# original architecture
class HufuNet(nn.Module):
    def __init__(self):
        super(HufuNet, self).__init__()
        # Encoder: compress 28x28 images to a latent representation of shape [64, 4, 4]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 20, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 20, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()

        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def encoder():
    return HufuNet()