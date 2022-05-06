# Necessary imports

import numpy as np
import matplotlib.pyplot as plt
import argparse

from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import LBFGS

from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import AerPauliExpectation
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector

import os
    
# Additional torch-related imports
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    NLLLoss,
    MaxPool2d,
    Flatten,
    Sequential,
    ReLU,
)
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # read args

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--specific_classes_names", type=str, nargs="+", default=[]) # by default (empty) select first 10
    parser.add_argument("--n_qubits", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_samples_train", type=int, default=100) # 500 is max for cifar # Set -1 to use all
    parser.add_argument("--n_samples_test", type=int, default=50)
    parser.add_argument("--n_samples_show", type=int, default=6)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    args = parser.parse_args()

    batch_size = args.batch_size
    n_samples_train = args.n_samples_train
    n_samples_test = args.n_samples_test
    n_samples_show = args.n_samples_show
    n_classes = args.n_classes
    n_qubits = args.n_qubits
    epochs = args.epochs
    LR = args.lr
    shuffle = args.shuffle
    specific_classes_names = args.specific_classes_names
    use_specific_classes = len(specific_classes_names)>=n_classes
    dataset = args.dataset
    print(specific_classes_names)
    
    # Set seed for random generators
    algorithm_globals.random_seed = args.seed
    manual_seed(args.seed) # Set train shuffle seed (for reproducibility)

    if not os.path.exists("plots"):
        os.mkdir("plots")

    ######## PREPARE DATASETS

    # Train Dataset
    # -------------
    
    # Use pre-defined torchvision function to load train data
    if dataset == "CIFAR100":
        classes_list = ['apple', 'fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        X_train = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
        )
        X_test = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
        )
    elif dataset == "MNIST":
        classes_list = ["0","1","2","3","4","5","6","7","8","9"]
        X_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
        )
        X_test = datasets.MNIST(
            root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
        )
    elif dataset == "CIFAR10":
        classes_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        X_train = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
        )
        X_test = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
        )

    # get conversion of identifiers for classes
    dict_classes = { class_name : i for i,class_name in enumerate(classes_list)}
    if not use_specific_classes:
        specific_classes = range(0, n_classes)
        specific_classes_names = list(map(str,specific_classes))
        #specific_classes_names = [classes_list[idx] for idx in specific_classes] ## old 
    else:
        specific_classes = [dict_classes[i] for i in specific_classes_names]

    classes_str = ",".join(specific_classes_names)
    classes2spec = {}
    for idx, class_idx in enumerate(specific_classes):
        classes2spec[class_idx]=idx

    # Filter out labels (originally 0-9), leaving only labels 0 and 1
    X_train.targets=np.int64(X_train.targets)

    n_samples = n_samples_train
    idx = np.int64([])    
    for i in range(0,n_classes):
        class_idx = specific_classes[i]
        idx = np.append(idx, np.where(X_train.targets == class_idx)[0][:n_samples])

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    # Define torch dataloader with filtered data
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=shuffle)

    # Get channels (rgb or grayscale)
    if len(X_train.data.shape)>3: # 3d image (rgb+)
        n_channels = X_train.data.shape[3]
        n_filts = 400
    else: # 2d image (grayscale)
        n_channels = 1
        n_filts = 256

        # Test Dataset
    # -------------

    # Set test shuffle seed (for reproducibility)
    # manual_seed(5)
    n_samples = n_samples_test

    # Filter out labels (originally 0-9), leaving only labels 0 and 1
    X_test.targets=np.int64(X_test.targets)

    idx = np.int64([])
    for i in range(0,n_classes):
        class_idx = specific_classes[i]
        idx = np.append(idx, np.where(X_test.targets == class_idx)[0][:n_samples])
    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]

    # Define torch dataloader with filtered data
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=shuffle)

    ##### VISUALIZE LABELS
    data_iter = iter(train_loader)
    n_samples_show_alt = n_samples_show
    while n_samples_show_alt > 0:
        images, targets = data_iter.__next__()
        plt.clf()
        fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(n_samples_show*2, 3))
        for idx, image in enumerate(images):
            axes[idx].imshow(np.moveaxis(images[idx].numpy().squeeze(),0,-1))
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            class_label = classes_list[targets[idx].item()]
            axes[idx].set_title("Labeled: {}".format(class_label))
            if idx > n_samples_show:
                plt.savefig(f"plots/{dataset}_classification{classes_str}_subplots{n_samples_show_alt}_lr{LR}_labeled_q{n_qubits}_{n_samples}samples_bsize{batch_size}_{epochs}epoch.png")
                break
        n_samples_show_alt -= 1

    ##### DESIGN NETWORK
    # declare quantum instance
    qi = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))

    # Define QNN
    feature_map = ZZFeatureMap(n_qubits)
    ansatz = RealAmplitudes(n_qubits, reps=1)
    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = TwoLayerQNN(
        n_qubits, feature_map, ansatz, input_gradients=True, exp_val=AerPauliExpectation(), quantum_instance=qi
    )
    print(qnn.operator)
    qnn.circuit.draw(output="mpl",filename=f"plots/qnn{n_qubits}_{n_classes}classes.png")
    #from qiskit.quantum_info import Statevector
    #from qiskit.visualization import plot_bloch_multivector
    #state = Statevector.from_instruction(qnn.circuit)
    #plot_bloch_multivector(state)

    class Net(Module):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv2d(n_channels, 2, kernel_size=5)
            self.conv2 = Conv2d(2, 16, kernel_size=5)
            self.dropout = Dropout2d()
            self.fc1 = Linear(n_filts, 64) # 256
            self.fc2 = Linear(64,n_qubits)  # 2-dimensional input to QNN
            self.qnn = TorchConnector(qnn)  # Apply torch connector, weights chosen
            # uniformly at random from interval [-1,1].
            if n_classes == 2:
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
            if n_classes == 2:
                return cat((x, 1 - x), -1)
            else:
                return x

    model4 = Net()
    print(model4)

    # Define model, optimizer, and loss function
    optimizer = optim.Adam(model4.parameters(), lr=LR)
    loss_func = NLLLoss()

    ################# TRAIN
    # Start training
    loss_list = []  # Store loss history
    model4.train()  # Set model to training mode

    for epoch in range(epochs):
        total_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)  # Initialize gradient
            output = model4(data)  # Forward pass

            # change target class identifiers towards 0 to n_classes
            for sample_idx, value in enumerate(target):
                target[sample_idx]=classes2spec[target[sample_idx].item()]

            loss = loss_func(output, target)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights
            total_loss.append(loss.item())  # Store loss
        loss_list.append(sum(total_loss) / len(total_loss))
        print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))

    # Plot loss convergence
    plt.clf()
    plt.plot(loss_list)
    plt.title("Hybrid NN Training Convergence")
    plt.xlabel("Training Iterations")
    plt.ylabel("Neg. Log Likelihood Loss")
    plt.savefig(f"plots/{dataset}_classification{classes_str}_hybridqnn_q{n_qubits}_{n_samples}samples_lr{LR}_bsize{batch_size}.png")
    
    ######## TEST
    model4.eval()  # set model to evaluation mode
    with no_grad():
        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model4(data)
            if len(output.shape) == 1:
                output = output.reshape(1, *output.shape)
            pred = output.argmax(dim=1, keepdim=True)

            # change target class identifiers towards 0 to n_classes
            for sample_idx, value in enumerate(target):
                target[sample_idx]=classes2spec[target[sample_idx].item()]
            
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = loss_func(output, target)
            total_loss.append(loss.item())

        print(
            "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
                sum(total_loss) / len(total_loss), correct / len(test_loader) / batch_size * 100
            )
        )

    # Plot predicted labels
    model4.eval()
    with no_grad():
        n_samples_show_alt = n_samples_show
        while n_samples_show_alt > 0:
            plt.clf()
            count = 0
            fig, axes = plt.subplots(nrows=1, ncols=n_samples_show*batch_size, figsize=(n_samples_show*2, batch_size*3))
            for batch_idx, (data, target) in enumerate(test_loader):    
                if count == n_samples_show:
                    plt.savefig(f"plots/{dataset}_classification{classes_str}_subplots{n_samples_show_alt}_lr{LR}_pred_q{n_qubits}_{n_samples}samples_bsize{batch_size}_{epochs}epoch.png")
                    break
                output = model4(data)
                if len(output.shape) == 1:
                    output = output.reshape(1, *output.shape)
                pred = output.argmax(dim=1, keepdim=True)
                for sample_idx in range(batch_size):
                    class_label = classes_list[specific_classes[pred[sample_idx].item()]]
                    axes[count].imshow(np.moveaxis(data[sample_idx].numpy().squeeze(),0,-1))
                    axes[count].set_xticks([])
                    axes[count].set_yticks([])
                    axes[count].set_title(class_label)
                    count += 1
            n_samples_show_alt -= 1
