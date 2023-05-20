from torch import nn

from train_utils import N_CLASSES

############################################################
# LeNet32 and LeNet128
############################################################


class LeNet32(nn.Module):
    IMAGE_CHANNELS = 3

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(  # 32 x 32
            nn.Conv2d(in_channels=self.IMAGE_CHANNELS, out_channels=6, kernel_size=5),  # 28 x 28
            nn.Tanh(),
            nn.MaxPool2d(2),  # 14 x 14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # 10 x 10
            nn.Tanh(),
            nn.MaxPool2d(2),  # 5 x 5
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)  # 1 x 1
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=N_CLASSES)
        )

    def forward(self, x):
        # x: B x 1 x TARGET_SIZE x TARGET_SIZE
        out = self.encoder(x)
        # out: B x 120 x 1 x 1
        out = out.squeeze(-1).squeeze(-1)
        # out: B x 120
        out = self.head(out)
        # out: B x 10
        return out


class LeNet128(nn.Module):
    IMAGE_CHANNELS = 3

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(  # 128 x 128
            nn.Conv2d(in_channels=self.IMAGE_CHANNELS, out_channels=3, kernel_size=5),  # 124 x 124
            nn.Tanh(),
            nn.MaxPool2d(2),  # 62 x 62
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),  # 58 x 58
            nn.Tanh(),
            nn.MaxPool2d(2),  # 29 x 29
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # 24 x 24
            nn.Tanh(),
            nn.MaxPool2d(2),  # 12 x 12
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5),  # 8 x 8
            nn.Tanh(),
            nn.MaxPool2d(2),  # 4 x 4
            nn.Conv2d(in_channels=64, out_channels=120, kernel_size=4),  # 1 x 1
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=N_CLASSES)
        )

    def forward(self, x):
        # x: B x 1 x TARGET_SIZE x TARGET_SIZE
        out = self.encoder(x)
        # out: B x 120 x 1 x 1
        out = out.squeeze(-1).squeeze(-1)
        # out: B x 120
        out = self.head(out)
        # out: B x 10
        return out
