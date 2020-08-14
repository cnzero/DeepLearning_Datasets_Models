import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


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