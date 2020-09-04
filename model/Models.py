import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from torch.nn.utils import weight_norm

class MNIST_LeNet(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class NinaPro_LeNet(BaseModel):
    def __init__(self, num_classes=52):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=6,
                               kernel_size=[9,3],
                               padding=2,
                               stride=1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2,
                                     stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=6)

        self.conv2 = nn.Conv2d(in_channels=6, 
                               out_channels=16,
                               kernel_size=[5,3],
                               padding=2,
                               stride=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, 
                                     stride=2)
        self.fc1 = nn.Linear(384, 160)
        self.fc2 = nn.Linear(160, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = torch.sigmoid(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.avgpool2(x)
        x = torch.sigmoid(x)
        # print('DEBUG x input shape: ', x.shape)
        x = x.view(size=(-1, x.shape[1]*x.shape[2]*x.shape[3]))
        # print('DEBUG x input shape: ', x.shape)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

class MNIST_AlexNet(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Fashion_AlexNet(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))  # 256*6*6 -> 4096
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, 0.5)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)

        return out


class NinaPro_instantaneousLeNet(BaseModel):
    def __init__(self,in_channels=10, num_classes=52):
        super().__init__()
        # - fake sample.shape = [m, 1, 10]
        sample = torch.randn(1, 1, in_channels)
        in_channels = sample.shape[1]

        # self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv1   = nn.Conv1d(in_channels,64,3,1)
        s = self.conv1(sample)

        # self.conv2 = nn.Conv2d(32,64,3,1)
        self.conv2   = nn.Conv1d(64,256,3,1)
        s = self.conv2(s)
        s = F.max_pool2d(s, 2)

        # self.dropout1 = nn.Dropout2d(0.25)
        self.dropout1   = nn.Dropout(0.5)

        # self.dropout2 = nn.Dropout2d(0.5)
        self.dropout2   = nn.Dropout(0.5)

        # self.fc1 = nn.Linear(9216, 128)
        # - x.shape = (m, out_channels, d)
        self.fc1   = nn.Linear(s.shape[1]*s.shape[2], 128) # maxpool=2, 64*3/2

        # self.fc2 = nn.Linear(128, 10)
        self.fc2   = nn.Linear(128, num_classes) # for DB1 0-12 gestures.

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)

        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)

        x = self.fc2(x)

        outputs = F.log_softmax(x, dim=1)
        return outputs



# original 'tcn' model.
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, 
                 n_inputs, 
                 n_outputs, 
                 kernel_size,
                 stride,
                 dilation, 
                 padding, 
                 dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, 
                                           kernel_size=kernel_size, 
                                           stride=stride, 
                                           padding=padding,
                                           dilation=dilation)
                                )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, 
                                           kernel_size=kernel_size,
                                           padding=padding,
                                           dilation=dilation)
                                )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # sigma = 0.01
        sigma = 100.00
        self.conv1.weight.data.normal_(0, sigma)
        self.conv2.weight.data.normal_(0, sigma)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, sigma)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i==0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, 
                                     kernel_size=kernel_size, 
                                     stride=1, 
                                     dilation=dilation_size,
                                     padding=(kernel_size-1)*dilation_size,
                                     dropout=dropout)
                      ]

            self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NinaPro_TCN_Net(BaseModel):
    def __init__(self, in_channels=10, num_classes=52, \
                 kernel_size=2, \
                 dropout=0.2):
        super(NinaPro_TCN_Net, self).__init__()
        # num_channels = [25, 25, 25, 25]
        num_channels = [[32, 32], [32, 32], [64, 64], [128, 128]]
        self.tcn = TemporalConvNet(num_inputs=in_channels,\
                                   num_channels=num_channels,\
                                   kernel_size=kernel_size, \
                                   dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], num_classes)
        

    def forward(self, x):
        # x.shape = [10608, 30, 10]
        # y.shape = [10608, 1]
        x = x.view((x.shape[0], x.shape[2], x.shape[1]))
        # x.shape = [m, LW*nChannels]
        y1 = self.tcn(x) # input should have dimension (N, C, L)
        output = self.linear(y1[:, :, -1])
        return F.log_softmax(output, dim=1)
