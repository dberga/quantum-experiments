# Necessary imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

from torch import Tensor

from qiskit import Aer, QuantumCircuit, BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import AerPauliExpectation
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector

from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.multiclass_extensions import AllPairs
from qiskit.aqua.utils.dataset_helper import get_feature_dimension

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from quantic.data import DatasetLoader
    
# Additional torch-related imports
from torch import no_grad, manual_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim # Adam, SGD, LBFGS
from torch.nn import NLLLoss # CrossEntropyLoss, MSELoss
from qnetworks import HybridQNN_Shallow

#time
import timeit


if __name__ == "__main__":
    
    # read args

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--specific_classes_names", type=str, nargs="+", default=[]) # by default (empty) select first 10
    parser.add_argument("--n_qubits", type=int, default=2)
    parser.add_argument("--n_features", type=int, default=None)
    parser.add_argument("--network", type=str, default="hybridqnn_shallow")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_samples_train", type=int, default=100) # 500 is max for cifar # Set -1 to use all
    parser.add_argument("--n_samples_test", type=int, default=50)
    parser.add_argument("--n_samples_show", type=int, default=6)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001))
    parser.add_argument("--input_res", type=int, default=28)
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--dataset_cfg", type=str, default="")
    args = parser.parse_args()

    # network args
    n_classes = args.n_classes
    n_qubits = args.n_qubits
    n_features = args.n_features
    if n_features is None:
        n_features = n_qubits
    network = args.network
    # train args
    batch_size = args.batch_size
    epochs = args.epochs
    LR = args.lr
    n_samples_train = args.n_samples_train
    n_samples_test = args.n_samples_test
    n_samples_show = args.n_samples_show
    # dataset args
    shuffle = args.shuffle
    specific_classes_names = args.specific_classes_names
    use_specific_classes = len(specific_classes_names)>=n_classes
    dataset = args.dataset
    dataset_cfg = args.dataset_cfg
    print(specific_classes_names)
    # preprocessing
    input_resolution = (args.input_res,args.input_res) #(28,28) # please check n_filts required on fc1/fc2 to input to qnn
    resize_interpolation = transforms.functional.InterpolationMode.BILINEAR 
    
    # Set seed for random generators
    algorithm_globals.random_seed = args.seed
    manual_seed(args.seed) # Set train shuffle seed (for reproducibility)

    if not os.path.exists("plots"):
        os.mkdir("plots")

    ######## PREPARE DATASETS
time_start = timeit.timeit()

# Train Dataset
# -------------

if dataset_cfg:
    # Load a dataset configuration file
    dataset = DatasetLoader.load(from_cfg=dataset_cfg,framework='torchvision')
else:
    # Or instantate dataset manually
    dataset = DatasetLoader.load(dataset_type=dataset,
                                   num_classes=n_classes,
                                   specific_classes=specific_classes_names,
                                   num_samples_class_train=n_samples_train,
                                   num_samples_class_test=n_samples_test,
                                   framework='torchvision'
                                   )
    print(f'Dataset partitions: {dataset.get_partitions()}')

    X_train = dataset['train']
    X_test = dataset['test']

    # Get channels (rgb or grayscale)
    if len(X_train.data.shape)>3: # 3d image (rgb+)
        n_channels = X_train.data.shape[3]
    else: # 2d image (grayscale)
        n_channels = 1

    # Set preprocessing transforms
    list_preprocessing = [
        transforms.Resize(input_resolution),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
    ] #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    X_train.transform= transforms.Compose(list_preprocessing)
    X_test.transform = transforms.Compose(list_preprocessing)           

    # set filtered/specific class names
    classes_str = ",".join(dataset.specific_classes_names)
    classes2spec = {}
    specific_classes = dataset.specific_classes
    for idx, class_idx in enumerate(specific_classes):
        classes2spec[class_idx]=idx

    classes_list = dataset.classes
    n_samples = n_samples_test

    # Previous dataset loading code
    '''
    # # # Use pre-defined torchvision function to load train data
    # # if dataset == "CIFAR100":
    # #     classes_list = ['apple', 'fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    # #     X_train = datasets.CIFAR100(
    # #         root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    # #     )
    # #     X_test = datasets.CIFAR100(
    # #         root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
    # #     )
    # # elif dataset == "MNIST":
    # #     classes_list = ["0","1","2","3","4","5","6","7","8","9"]
    # #     X_train = datasets.MNIST(
    # #         root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    # #     )
    # #     X_test = datasets.MNIST(
    # #         root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
    # #     )
    # # elif dataset == "CIFAR10":
    # #     classes_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    # #     X_train = datasets.CIFAR10(
    # #         root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    # #     )
    # #     X_test = datasets.CIFAR10(
    # #         root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
    # #     )

    # # # get conversion of identifiers for classes
    # # dict_classes = { class_name : i for i,class_name in enumerate(classes_list)}
    # # if not use_specific_classes:
    # #     specific_classes = range(0, n_classes)
    # #     specific_classes_names = list(map(str,specific_classes))
    # #     #specific_classes_names = [classes_list[idx] for idx in specific_classes] ## old 
    # # else:
    # #     specific_classes = [dict_classes[i] for i in specific_classes_names]

    # # classes_str = ",".join(specific_classes_names)
    # # classes2spec = {}
    # # for idx, class_idx in enumerate(specific_classes):
    # #     classes2spec[class_idx]=idx

    # # # Filter out labels (originally 0-9), leaving only labels 0 and 1
    # # X_train.targets=np.int64(X_train.targets)

    # # n_samples = n_samples_train
    # # idx = np.int64([])    
    # # for i in range(0,n_classes):
    # #     class_idx = specific_classes[i]
    # #     idx = np.append(idx, np.where(X_train.targets == class_idx)[0][:n_samples])

    # # X_train.data = X_train.data[idx]
    # # X_train.targets = X_train.targets[idx]

    # # # Test Dataset
    # # # -------------

    # # # Set test shuffle seed (for reproducibility)
    # # # manual_seed(5)
    # # n_samples = n_samples_test

    # # # Filter out labels (originally 0-9), leaving only labels 0 and 1
    # # X_test.targets=np.int64(X_test.targets)

    # # idx = np.int64([])
    # # for i in range(0,n_classes):
    # #     class_idx = specific_classes[i]
    # #     idx = np.append(idx, np.where(X_test.targets == class_idx)[0][:n_samples])
    # # X_test.data = X_test.data[idx]
    # # X_test.targets = X_test.targets[idx]
    '''
    
    # Define torch dataloader with filtered data
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=shuffle)
    # Define torch dataloader with filtered data
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=shuffle)

    time_end = timeit.timeit()
    time_elapsed = time_end - time_start
    print(f"Dataloaders elapsed time: {time_elapsed} s")

    ##### VISUALIZE LABELS
    data_iter = iter(train_loader)
    n_samples_show_alt = n_samples_show
    while n_samples_show_alt > 0:
        try:
            images, targets = data_iter.__next__()
        except:
            break
        plt.clf()
        fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(n_samples_show*2, batch_size*3))
        if n_samples_show == 1:
            axes = [axes]
        for idx, image in enumerate(images):
            axes[idx].imshow(np.moveaxis(images[idx].numpy().squeeze(),0,-1))
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            class_label = classes_list[targets[idx].item()]
            axes[idx].set_title("Labeled: {}".format(class_label))
            if idx > n_samples_show: 
                plt.savefig("plots/{dataset}_classification{classes_str}_subplots{n_samples_show_alt}_lr{LR}_labeled_q{n_qubits}_{n_samples}samples_bsize{batch_size}_{epochs}epoch.png")
                break
        plt.show()
        n_samples_show_alt -= 1
    

    ##### DESIGN NETWORK
    time_start = timeit.timeit()

    # init network
    if network == "hybridqnn_shallow":
        ## predefine number of input filters to fc1 and fc2
        # examples: 13456 for (128x128x1), 59536 for (256x256x1), 256 for (28x28x1), 400 for (28x28x3) or (35x35x1) 
        if n_channels == 1:
            n_filts_fc1 = int(((((input_resolution[0]-4)/2)-4)/2)**2)*16
        else:
            n_filts_fc1 = int(((((input_resolution[0]+7-4)/2)-4)/2)**2)*16
        n_filts_fc2 = int(n_filts_fc1 / 4)

        # declare quantum instance
        qi = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))
        # Define QNN
        feature_map = ZZFeatureMap(n_features)
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
        model = HybridQNN_Shallow(n_classes = n_classes, n_qubits = n_qubits, n_channels = n_channels, n_filts_fc1 = n_filts_fc1, n_filts_fc2 = n_filts_fc2, qnn = qnn)
        print(model)

        # Define model, optimizer, and loss function
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_func = NLLLoss()

    elif network == "QSVM":
        backend = BasicAer.get_backend('qasm_simulator')

        # todo: fix this transformation for QSVM
        train_input = X_train.targets
        test_input = X_test.targets
        total_array = np.concatenate([test_input[k] for k in test_input])
        #
        feature_map = ZZFeatureMap(feature_dimension=get_feature_dimension(train_input),
                                   reps=2, entanglement='linear')
        svm = QSVM(feature_map, train_input, test_input, total_array,
                   multiclass_extension=AllPairs())
        quantum_instance = QuantumInstance(backend, shots=1024,
                                           seed_simulator=algorithm_globals.random_seed,
                                           seed_transpiler=algorithm_globals.random_seed)
    else:
        model = HybridQNN(backbone=network,pretrained=True,n_qubits=n_qubits,n_classes=n_classes)
        # Define model, optimizer, and loss function
        optimizer = optim.Adam(model.network.parameters(), lr=LR)
        loss_func = NLLLoss()

    time_end = timeit.timeit()
    time_elapsed = time_end - time_start
    print(f"Network init elapsed time: {time_elapsed} s")
    ################# TRAIN
    # Start training
    time_start = timeit.timeit()

    if network != "QSVM":
        if "vgg" in network or "resnet" in network:
            model.network.train()  # Set model to training mode
        if network == "hybridqnn_shallow":
            model.train()  # Set model to training mode

        loss_list = []  # Store loss history
        for epoch in range(epochs):
            total_loss = []
            for batch_idx, (data, target) in enumerate(train_loader):

                optimizer.zero_grad(set_to_none=True)  # Initialize gradient
                output = model(data)  # Forward pass

                # change target class identifiers towards 0 to n_classes
                for sample_idx, value in enumerate(target):
                    target[sample_idx]=classes2spec[target[sample_idx].item()]

                loss = loss_func(output, target)  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Optimize weights
                total_loss.append(loss.item())  # Store loss
                print(f"Batch {batch_idx}, Loss: {total_loss[-1]}")
            loss_list.append(sum(total_loss) / len(total_loss))
            print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))

        # Plot loss convergence
        plt.clf()
        plt.plot(loss_list)
        plt.title("Hybrid NN Training Convergence")
        plt.xlabel("Training Iterations")
        plt.ylabel("Neg. Log Likelihood Loss")
        plt.savefig(f"plots/{dataset}_classification{classes_str}_hybridqnn_q{n_qubits}_{n_samples}samples_lr{LR}_bsize{batch_size}.png")
        plt.show()

    elif network == "QSVM":
        result = svm.run(quantum_instance)
        for k,v in result.items():
            print(f'{k} : {v}')

    time_end = timeit.timeit()
    time_elapsed = time_end - time_start
    print(f"Training time: {time_elapsed} s")

    ######## TEST
    time_start = timeit.timeit()

    if network != "QSVM":
        if "vgg" in network or "resnet" in network:
            model.network.eval()  # Set model to eval mode
        if network == "hybridqnn_shallow":
            model.eval()  # Set model to eval mode
        with no_grad():
            correct = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                output = model(data)
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

        time_end = timeit.timeit()
        time_elapsed = time_end - time_start
        print(f"Test time: {time_elapsed} s")
    
        # Plot predicted labels
        n_samples_show_alt = n_samples_show
        while n_samples_show_alt > 0:
            plt.clf()
            count = 0
            fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(n_samples_show*2, batch_size*3))
            if n_samples_show == 1:
                axes = [axes]
            for batch_idx, (data, target) in enumerate(test_loader):    
                if count == n_samples_show:
                    plt.savefig(f"plots/{dataset}_classification{classes_str}_subplots{n_samples_show_alt}_lr{LR}_pred_q{n_qubits}_{n_samples}samples_bsize{batch_size}_{epochs}epoch.png")
                    plt.show()
                    break
                output = model(data)
                if len(output.shape) == 1:
                    output = output.reshape(1, *output.shape)
                pred = output.argmax(dim=1, keepdim=True)
                for sample_idx in range(batch_size):
                    try:
                        class_label = classes_list[specific_classes[pred[sample_idx].item()]]
                    except:
                        class_label = classes_list[specific_classes[pred[sample_idx].item()-1]]
                    axes[count].imshow(np.moveaxis(data[sample_idx].numpy().squeeze(),0,-1))
                    axes[count].set_xticks([])
                    axes[count].set_yticks([])
                    axes[count].set_title(class_label)
                    count += 1
            n_samples_show_alt -= 1
