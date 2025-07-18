import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, dilation_list):
        super().__init__()
        self.branches = nn.ModuleList()

        for d in dilation_list:
            branch = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels,
                    in_channels, 
                    kernel_size=3, 
                    padding='same',
                    dilation=d
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels,
                    in_channels, 
                    kernel_size=3, 
                    padding='same',
                    dilation=d
                )
            )
            
            self.branches.append(branch)

    def forward(self, x):
        output = x
        
        for branch in self.branches:
            output = output + branch(x)
        
        return output
    

class DownSampleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_list):
        super().__init__()
        self.res_block = ResBlock(in_channels, dilation_list)
        self.down_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=2
        )

    def forward(self, x):
        x = self.res_block(x)
        return self.down_conv(x)
    

class UpSampleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_list):
        super().__init__()
        self.res_block = ResBlock(in_channels, dilation_list)
        self.up_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(self, x):
        x = self.res_block(x)
        return self.up_conv(x)
    

class Combine(nn.Module):
    def __init__(self, in_channels, out_channels):
        # in_channels: x1.features + x2.features
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1, 
            dilation=1,
            stride=1
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x1, x2):
        x1 = self.relu(x1)
        x = torch.cat([x1, x2], 1)
        x = self.conv(x)
        return self.bn(x)


class PSPPooling(nn.Module):
    """
    Reference:
    Pyramid Scene Parsing Network
    https://arxiv.org/pdf/1612.01105
    """
    def __init__(self, in_channels, scale_factors=[1, 2, 4, 8]):
        # size: (b, 1024, 8, 8)
        super().__init__()  
        self.branches = nn.ModuleList()

        for scale in scale_factors:
            branch = nn.Sequential(
                nn.MaxPool2d(scale), # in_channels: 1024
                nn.Conv2d(
                    in_channels,
                    in_channels // len(scale_factors), 
                    kernel_size=1
                ),
                nn.Upsample(scale_factor=scale)
            )
            self.branches.append(branch)

        self.conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        output_stack = [x]

        for branch in self.branches:
            output_stack.append(branch(x))

        output = torch.cat(output_stack, 1)

        return self.conv(output)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = ResBlock(64, dilation_list=[1])
    output_tensor = model(input_tensor)
    print(input_tensor.shape)
    print(output_tensor.shape)

    combine = Combine(3 * 2, 4)
    x1 = torch.randn(1, 3, 64, 64) 
    x2 = torch.randn(1, 3, 64, 64)
    output = combine(x1, x2)
    print(output.shape)

    psp_pooling = PSPPooling(64)
    input_tensor = torch.randn(1, 64, 8, 8) # smallest size is 8x8
    output = psp_pooling(input_tensor)
    print(output.shape)