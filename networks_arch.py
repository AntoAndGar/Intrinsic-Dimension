import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

# Class for a Fully Connected Network
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(FullyConnectedNetwork, self).__init__()
        self.num_layers = num_layers
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        if num_layers > 0:
            self.fcs = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
            )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc_in(x))
        if self.num_layers > 0:
            for fc in self.fcs:
                x = F.relu(fc(x))
        x = self.fc_out(x)
        return x


# Class for Standard LeNet Network, reference http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf with some modification
# to follow the same number of parameters as the "Mesuring Intrinsic Dimensionality of Objective Landscape" paper for the task does
class LeNet(nn.Module):
    def __init__(self, input_dim, output_dim, dataset="mnist"):
        super(LeNet, self).__init__()
        # 6 kernels 5x5
        self.conv1 = nn.Conv2d(
            input_dim,
            6,
            5,
            padding="valid",
        )
        # max-pooling over 2x2
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # 16 kernels 5x5
        self.conv2 = nn.Conv2d(6, 16, 5, padding="valid")
        # max-pooling over 2x2
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # 120 kernels 4x4 to match the dimensionality of the fully connected network
        self.conv3 = nn.Conv2d(
            16,
            120,
            5 if dataset == "cifar10" else 4,
        )
        # 120 fully connected neurons, too many parameter in this case w.r.t. the paper
        # self.fc1 = nn.Linear(16 * 5 * 5, 120,)
        self.flat = nn.Flatten(start_dim=1)
        # 84 fully connected neurons
        self.fc2 = nn.Linear(120, 84)
        # 10 fully connected neurons
        self.fc3 = nn.Linear(
            84,
            output_dim,
        )

    def forward(self, x):
        # x = x.view(-1, 1, 28, 28)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.flat(x)
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# from https://discuss.pytorch.org/t/locally-connected-layers/26979
class LocallyConnected2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, output_size, kernel_size, stride=1, bias=True
    ):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            nn.init.kaiming_normal_(
                torch.randn(
                    1,
                    out_channels,
                    in_channels,
                    output_size[0],
                    output_size[1],
                    kernel_size**2,
                ),
                nonlinearity="relu",
            )
        )
        if bias:
            self.bias = nn.Parameter(
                nn.init.kaiming_normal_(
                    torch.randn(1, out_channels, output_size[0], output_size[1]),
                    nonlinearity="relu",
                )
            )
        else:
            self.register_parameter("bias", None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        #_, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


# Class for Untied LeNet Network
class Untied_LeNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Untied_LeNet, self).__init__()
        # 6 kernels 5x5, output size 24x24
        self.conv1 = LocallyConnected2d(input_dim, 6, (24, 24), 5)
        # max-pooling over 2x2
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # 16 kernels 5x5 output size 8x8
        self.conv2 = LocallyConnected2d(6, 16, (8, 8), 5)
        # max-pooling over 2x2
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # 120 kernels 4x4 to match the dimensionality of the fully connected network and obtain an output size of 1x1
        self.conv3 = LocallyConnected2d(16, 120, (1, 1), 4)
        # 120 fully connected neurons, too many parameter in this case w.r.t. the paper
        # self.fc1 = nn.Linear(16 * 5 * 5, 120,)
        self.flat = nn.Flatten(start_dim=1)
        # 84 fully connected neurons
        self.fc2 = nn.Linear(120, 84)
        # 10 fully connected neurons
        self.fc3 = nn.Linear(
            84,
            output_dim,
        )

    def forward(self, x):
        # x = x.view(-1, 1, 28, 28)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.flat(x)
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Class for FC-LeNet Network
class FcLeNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FcLeNet, self).__init__()
        # 
        self.fcconv1 = nn.Linear(input_dim, 3456)
        # max-pooling over 2x2
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # 16 kernels 5x5
        self.fcconv2 = nn.Linear(864, 1024)
        # max-pooling over 2x2
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # 120 kernels 4x4 to match the dimensionality of the fully connected network
        self.fcconv3 = nn.Linear(256, 120)
        # 120 fully connected neurons, too many parameter in this case w.r.t. the paper
        # self.fc1 = nn.Linear(16 * 5 * 5, 120,)
        self.flat = nn.Flatten(start_dim=1)
        # 84 fully connected neurons
        self.fc2 = nn.Linear(120, 84)
        # 10 fully connected neurons
        self.fc3 = nn.Linear(
            84,
            output_dim,
        )

    def forward(self, x):
        x = self.flat(x)
        x = self.pool1(F.relu(self.fcconv1(x)).view(-1, 6, 24, 24))
        x = self.flat(x)
        x = self.pool2(F.relu(self.fcconv2(x)).view(-1, 16, 8, 8))
        x = self.flat(x)
        x = F.relu(self.fcconv3(x))
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Class for FCTied-LeNet
class FCTied_LeNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCTied_LeNet, self).__init__()
        # 6 kernels 5x5
        self.conv1 = nn.Conv2d(
            input_dim,
            6,
            55,
            padding="same",
        )
        # max-pooling over 2x2
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # 16 kernels 5x5
        self.conv2 = nn.Conv2d(6, 16, 27, padding="same")
        # max-pooling over 2x2
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # 120 kernels 4x4 to match the dimensionality of the fully connected network
        self.conv3 = nn.Conv2d(
            16,
            120,
            7,
        )
        # 120 fully connected neurons, too many parameter in this case w.r.t. the paper
        # self.fc1 = nn.Linear(16 * 5 * 5, 120,)
        self.flat = nn.Flatten(start_dim=1)
        # 84 fully connected neurons
        self.fc2 = nn.Linear(120, 84)
        # 10 fully connected neurons
        self.fc3 = nn.Linear(
            84,
            output_dim,
        )

    def forward(self, x):
        #x = x.view(-1, 1, 28, 28)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.flat(x)
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
