# Image classification

Detect specific categories from an input image

---

## MNIST: Hand-written digit recognition

Dataset with 10 numeric hand-written digit categories (0-9)
[[ref](http://yann.lecun.com/exdb/mnist/)]

### Datasets

| Dataset                                                             | Description                                                                                        |
| ---                                                                 | ---                                                                                                |
| [mnist_0_1_small.json](mnist_3_6_small.json)                        | Classify between digit "0" and "1". 100 images per class for training and 50 for testing.          |
| [mnist_3_6_small.json](mnist_3_6_small.json)                        | Classify between digit "3" and "6". 100 images per class for training and 50 for testing.          |
| [mnist_0-9_medium.json](mnist_0-9_medium.json)                        | Classify between all digits. 500 images per class for training and 100 for testing.          |

### Experiments

| Dataset config                                                     | Method                                                          |  Accuracy |
| ---                                                                | ---                                                             | ---       |
| [mnist_0_1_small.json](mnist_0_1_small.json)                       | [HybridQNN](/qiskit_classification/classification_hybridqnn.py) |    100.%  |
| [mnist_0_1_small.json](mnist_0_1_small.json)                       | [QSVM](/qiskit_classification/QSVM/Quantum_SVM_MNIST_0_1.ipynb)                                                       |    96 %  |
| [mnist_3_6_small.json](mnist_3_6_small.json)                       | [HybridQNN](/qiskit_classification/classification_hybridqnn.py) |    100.%  |
| [mnist_3_6_small.json](mnist_3_6_small.json)                       | [QSVM](/qiskit_classification/QSVM/Quantum_SVM_MNIST_3_6.ipynb)                                                       |    90 %  |
| [mnist_0-9_medium.json](mnist_0-9_medium.json)                       | [QSVM](/qiskit_classification/QSVM/Quantum_SVM_MNIST_digits_0_to_9.ipynb)                                                       |    52.3 %  |
---

## CIFAR: Image recognition

Classification amongst 10 different categories (CIFAR-10) or between 100 categories (CIFAR-100)
[[ref](https://www.cs.toronto.edu/~kriz/cifar.html)]

### Datasets

| Dataset                                                             | Description                                                                                        |
| ---                                                                 | ---                                                                                                |
| [cifar10_airplane_dog_small.json](cifar10_airplane_dog_small.json)  | Classify between digit "airplane" and "dog". 100 images per class for training and 50 for testing. |

### Experiments

| Dataset config                                                     | Method                                                          |  Accuracy |
| ---                                                                | ---                                                             | ---       |
| [cifar10_airplane_dog_small.json](cifar10_airplane_dog_small.json) | [HybridQNN](/qiskit_classification/classification_hybridqnn.py) |    50.0%  |
| [cifar10_airplane_dog_small.json](cifar10_airplane_dog_small.json) | [QSVM](/qiskit_classification/QSVM/Quantum_SVM_MNIST_airplane_dog.ipynb)                                                       |    63 %  |
