"""CNN architecture for CIFAR-10 image classification.

Input : 3 x 32 x 32 RGB images
Output: 10 class logits
"""
import torch.nn as nn


class CIFARNet(nn.Module):
    """A compact VGG-style CNN: three conv blocks + a classifier head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def conv_block(in_ch, out_ch):
            # Two 3x3 convs, batch-norm, ReLU, then halve the spatial size.
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),    # 32x32 -> 16x16
            conv_block(32, 64),   # 16x16 -> 8x8
            conv_block(64, 128),  # 8x8  -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
