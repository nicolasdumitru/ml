# https://arxiv.org/abs/1506.02640

from torch import nn


class Yolo(nn.Module):
    def __init__(self, grid_size, num_boxes, num_classes):
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.sequential = nn.Sequential(
            # 1. Convolve & maxpool (N, 3, 448, 448) -> (N, 64, 112, 112)
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            self.leaky_relu,
            self.maxpool,
            # 2. Convolve & maxpool (N, 64, 112, 112) -> (N, 192, 56, 56)
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.Conv2d(64, 192, 3, padding="same"),
            self.leaky_relu,
            self.maxpool,
            # 3. Conv. block & maxpool (N, 192, 56, 56) -> (N, 512, 28, 28)
            nn.Conv2d(192, 128, 1, padding="same"),
            self.leaky_relu,
            nn.Conv2d(128, 256, 3, padding="same"),
            self.leaky_relu,
            nn.Conv2d(256, 256, 1, padding="same"),
            self.leaky_relu,
            nn.Conv2d(256, 512, 3, padding="same"),
            self.leaky_relu,
            self.maxpool,
            # 4. Conv. block & maxpool (N, 512, 28, 28) -> (N, 1024, 14, 14)
            *[
                nn.Sequential(
                    nn.Conv2d(512, 256, 1, padding="same"),
                    self.leaky_relu,
                    nn.Conv2d(256, 512, 3, padding="same"),
                    self.leaky_relu,
                )
                for _ in range(4)
            ],
            nn.Conv2d(512, 512, 1, padding="same"),
            self.leaky_relu,
            nn.Conv2d(512, 1024, 3, padding="same"),
            self.leaky_relu,
            self.maxpool,
            # 5. Conv. block (N, 1024, 14, 14) -> (N, 1024, 7, 7)
            *[
                nn.Sequential(
                    nn.Conv2d(1024, 512, 1, padding="same"),
                    self.leaky_relu,
                    nn.Conv2d(512, 1024, 3, padding="same"),
                    self.leaky_relu,
                )
                for _ in range(2)
            ],
            nn.Conv2d(1024, 1024, 3, padding="same"),
            self.leaky_relu,
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            self.leaky_relu,
            # 6. Conv. block (N, 1024, 7, 7) -> (N, 1024, 7, 7)
            nn.Conv2d(1024, 1024, 3, padding="same"),
            self.leaky_relu,
            nn.Conv2d(1024, 1024, 3, padding="same"),
            self.leaky_relu,
            # 7. Affine linear transformations with dropout regularization
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            self.leaky_relu,
            self.dropout,
            nn.Linear(
                4096,
                self.grid_size
                * self.grid_size
                * (self.num_boxes * 5 + self.num_classes),
            )
        )

    def forward(self, x):
        return self.sequential(x).reshape(
            (-1, self.grid_size, self.grid_size, self.num_boxes * 5 + self.num_classes)
        )
