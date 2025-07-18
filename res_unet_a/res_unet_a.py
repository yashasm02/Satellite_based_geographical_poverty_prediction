import torch 
from torch import nn

from .modules import (
    Combine,
    PSPPooling,
    ResBlock,
    DownSampleResBlock,
    UpSampleResBlock
)


class ResUNetA(nn.Module):
    """ResUNet-a d6 single-task model

    ResUNet + Atrous + PSP
    input_shape: (b, c, >= 512, >= 512)
    
    Reference:
    ResUNet-a: A deep learning framework for semantic segmentation 
    of remotely sensed data
    https://doi.org/10.1016/j.isprsjprs.2020.01.013
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=1), # layer 1
            ResBlock(32, dilation_list=[1, 3, 15, 31]), # 2
            DownSampleResBlock(32, 64, dilation_list=[1, 3, 15, 31]), # 3 - 4
            DownSampleResBlock(64, 128, dilation_list=[1, 3, 15]), # 5 - 6
            DownSampleResBlock(128, 256, dilation_list=[1, 3, 15]), # 7 - 8
            DownSampleResBlock(256, 512, dilation_list=[1]), # 9 - 10
            DownSampleResBlock(512, 1024, dilation_list=[1] )# 11 - 12
        ])

        self.bottleneck = PSPPooling(1024) # 13

        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2), # 14
            Combine(512 * 2, 512), # 15
            UpSampleResBlock(512, 256, dilation_list=[1]), # 16 - 17
            Combine(256 * 2, 256), # 18
            UpSampleResBlock(256, 128, dilation_list=[1, 3, 15]), # 19 - 20
            Combine(128 * 2, 128), # 21
            UpSampleResBlock(128, 64, dilation_list=[1, 3, 15]), # 22 - 23
            Combine(64 * 2, 64), # 24
            UpSampleResBlock(64, 32, dilation_list=[1, 3, 15, 31]), # 25 - 26
            Combine(32 * 2, 32), # 27
            ResBlock(32, dilation_list=[1, 3, 15, 31]), # 28
            Combine(32 * 2, 32), # 29
            PSPPooling(32), # 30
        ])
        
        self.out = nn.Conv2d(32, out_channels, kernel_size=1, stride=1) # 31

    def forward(self, x):
        encoder_output_stack = []
        p = len(encoder_output_stack) - 2

        for layer in self.encoder:
            x = layer(x)   
            encoder_output_stack.append(x)

        x = self.bottleneck(x)

        for i, layer in enumerate(self.decoder):
            if i % 2:
                x = layer(x, encoder_output_stack[p])
                p -= 1
            else:
                x = layer(x)

        return self.out(x)
    

if __name__ == '__main__':
    model = ResUNetA(3, 2)
    x = torch.randn(1, 3, 512, 512)
    out = model(x)

    print(x.shape)
    print(out.shape)
    print(sum(p.numel() for p in model.parameters()))