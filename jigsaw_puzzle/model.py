import torch
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(
        self,
        in_c: int,
        pieces: int,
        image_size: int,
        hid_c: int = 64,
        stride: int = 2,
        kernel_size: int = 5
    ):
        super().__init__()
        self.g_1 = nn.Sequential(
            nn.Conv2d(in_c, hid_c, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.MaxPool2d(stride),
        )
        self.g_2 = nn.Linear(image_size**2//(stride**2*pieces**2)*64, pieces**2, bias=False)

        nn.init.kaiming_normal_(self.g_1[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.g_1[0].bias, 0)
        nn.init.kaiming_normal_(self.g_2.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, pieces):
        batch_size = pieces.shape[0]
        pieces = pieces.transpose(0,1).contiguous() # (num_pieces, batchsize, channels, height, width)
        pieces = [self.g_1(p).reshape(batch_size, -1) for p in pieces] # convolve and vectorize
        latent = [self.g_2(p) for p in pieces]
        latent_matrix = torch.stack(latent, 1)
        return latent_matrix

class Vectorize(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        return x.reshape(x.shape[0], self.channels)
