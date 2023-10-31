import torch.nn as nn
import torch.nn.functional as F


class GetShape(nn.Module):
    def forward(self, input):
        print(input.shape)    
        return input

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Unflatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), input.size(1), 1, 1)

class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

# decoder block (used in the decoder)
class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double the dimensions
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2, output_padding=1,
                                       bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, z_size, channel_in=1):
        super(Encoder, self).__init__()
        self.size = channel_in
        self.z_dim = z_size

        layers_list = []
        # the first time 3->64, for every other double the channel size
        for i in range(5):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=64)) #64
                self.size = 64
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2

        # final shape Bx256x8x8
        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=8 * 8 * self.size, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024,  momentum=0.9),
                                nn.ReLU(),
                               # nn.BatchNorm1d(num_features=1024,  momentum=0.9),
                                )
        #self.dropout = nn.Dropout2d(0.3)


    def forward(self, x):
        x = self.conv(x)
        x = x.view(len(x), -1)
        x = self.fc(x)
        return x

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


class Decoder(nn.Module):
    def __init__(self, z_size, size):
        super(Decoder, self).__init__()
        # start from B*z_size
        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=8 * 8 * size, bias=False),
                                nn.BatchNorm1d(num_features=8 * 8 * size,momentum=0.9),
                                nn.ReLU())
        self.size = size
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size))
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size//2))
        self.size = self.size//2
        layers_list.append(nn.Dropout2d(0.3))
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size//4))
        self.size = self.size//4
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size//8))
        self.size = self.size//8
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size//16))
        self.size = self.size//16
        # final conv to get 3 channels and tanh layer
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1),
            nn.Sigmoid() # try log sigmoid
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(len(x), -1, 8, 8)
        x = self.conv(x)
        return x

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)