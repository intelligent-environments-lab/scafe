import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):

    def __init__(self, latent_space_dim: int = 128):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=3,
                                out_channels=32,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.bn_1 = nn.BatchNorm2d(num_features=32)
        self.prelu_1 = nn.PReLU()

        self.conv_2 = nn.Conv2d(in_channels=32,
                                out_channels=64,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.bn_2 = nn.BatchNorm2d(num_features=64)
        self.prelu_2 = nn.PReLU()

        self.conv_3 = nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.bn_3 = nn.BatchNorm2d(num_features=128)
        self.prelu_3 = nn.PReLU()

        self.conv_4 = nn.Conv2d(in_channels=128,
                                out_channels=256,
                                kernel_size=3,
                                stride=1,
                                padding=0)
        self.bn_4 = nn.BatchNorm2d(num_features=256)
        self.prelu_4 = nn.PReLU()

        self.fc_out = nn.Linear(in_features=256, out_features=latent_space_dim)

        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu_1(self.bn_1(self.conv_1(x)))
        x = self.prelu_2(self.bn_2(self.conv_2(x)))
        x = self.prelu_3(self.bn_3(self.conv_3(x)))
        h = x.register_hook(self.activations_hook)
        x = self.prelu_4(self.bn_4(self.conv_4(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc_out(x)
        return x

    def extract_features(self, x):
        x = self.prelu_1(self.bn_1(self.conv_1(x)))
        x = self.prelu_2(self.bn_2(self.conv_2(x)))
        x = self.prelu_3(self.bn_3(self.conv_3(x)))
        x = self.prelu_4(self.bn_4(self.conv_4(x)))
        return x

    def get_activations(self, x):
        x = self.prelu_1(self.bn_1(self.conv_1(x)))
        x = self.prelu_2(self.bn_2(self.conv_2(x)))
        x = self.prelu_3(self.bn_3(self.conv_3(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, latent_space_dim: int = 128):
        super(Decoder, self).__init__()

        self.fc_in = nn.Linear(in_features=latent_space_dim, out_features=256)
        self.prelu_in = nn.PReLU()

        self.conv_1 = nn.ConvTranspose2d(in_channels=256,
                                         out_channels=128,
                                         kernel_size=3,
                                         stride=1,
                                         padding=0)
        self.bn_1 = nn.BatchNorm2d(num_features=128)
        self.prelu_1 = nn.PReLU()

        self.conv_2 = nn.Conv2d(in_channels=128,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.bn_2 = nn.BatchNorm2d(num_features=64)
        self.prelu_2 = nn.PReLU()

        self.conv_3 = nn.Conv2d(in_channels=64,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.bn_3 = nn.BatchNorm2d(num_features=32)
        self.prelu_3 = nn.PReLU()

        self.conv_4 = nn.Conv2d(in_channels=32,
                                out_channels=3,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.bn_4 = nn.BatchNorm2d(num_features=3)
        self.prelu_4 = nn.PReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.prelu_in(self.fc_in(z))
        x = x.view(x.shape[0], 256, 1, 1)
        x = self.prelu_1(self.bn_1(self.conv_1(x)))
        x = F.interpolate(self.prelu_2(self.bn_2(self.conv_2(x))), scale_factor=2)
        x = F.interpolate(self.prelu_3(self.bn_3(self.conv_3(x))), scale_factor=2)
        x = F.interpolate(self.prelu_4(self.bn_4(self.conv_4(x))), scale_factor=2)
        return x
