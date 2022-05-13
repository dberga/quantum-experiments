from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    MaxPool2d,
    Flatten,
    Sequential,
    ReLU,
)
from torch import cat
import torch.nn.functional as F
from qiskit_machine_learning.connectors import TorchConnector

class HybridQNN_Shallow(Module):
    def __init__(self, n_classes = 2, n_qubits = 2, n_channels = 3, n_filts = 400, qnn = None):
         
        # save for when computing head
        self.n_classes = n_classes

        super().__init__()
        self.conv1 = Conv2d(n_channels, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(n_filts, 64) # 256
        self.fc2 = Linear(64,n_qubits)  # 2-dimensional input to QNN
        self.qnn = TorchConnector(qnn)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        if self.n_classes == 2:
            self.fc3 = Linear(1, 1) # 1-dimensional output from QNN
        else:
            self.fc3 = Linear(1, n_classes) 
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)  # apply QNN
        x = self.fc3(x)
        if self.n_classes == 2:
            return cat((x, 1 - x), -1)
        else:
            return x
