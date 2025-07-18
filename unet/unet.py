import torch 
from torch import nn

from .modules import DoubleConv, DownSample, UpSample


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.ModuleList([
            DownSample(in_channels, 64),
            DownSample(64, 128),
            DownSample(128, 256),
            DownSample(256, 512)
        ])

        self.bottleneck = DoubleConv(512, 1024)

        self.decoder = nn.ModuleList([
            UpSample(1024, 512),
            UpSample(512, 256),
            UpSample(256, 128),
            UpSample(128, 64)
        ])

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        encoder_output = []

        for layer in self.encoder:
            x, x1 = layer(x)
            encoder_output.append(x1)
        
        x = self.bottleneck(x)

        for i, layer in enumerate(self.decoder):
            x = layer(x, encoder_output[~i])

        return self.output(x)
    

if __name__ == '__main__':
    x = torch.rand((1, 3, 512, 512))
    model = UNet(3, 10)
    output = model(x)
    print(output.shape)