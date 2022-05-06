"""
TF-Quantum tutorial: MNIST (https://www.tensorflow.org/quantum/tutorials/mnist)

Modifications:
- Use any pair of gt classes (instead of fixed 3 abnd 6)
- Phase encoding instead of binary
- Removed full CNN model comparison, only with fair-CNN 
"""
import importlib, pkg_resources
importlib.reload(pkg_resources)

import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import seaborn as sns
import collections

# visualization tools
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

# Dataset loading
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))

# Filter classes
def filter_gt_classes(x, y, class1, class2):
    keep = (y == class1) | (y == class2)
    x, y = x[keep], y[keep]
    y = y == class1
    return x,y

gt_class1, gt_class2 = 3, 6

x_train, y_train = filter_gt_classes(x_train, y_train, gt_class1, gt_class2)
x_test, y_test = filter_gt_classes(x_test, y_test, gt_class1, gt_class2)

print("Number of filtered training examples:", len(x_train))
print("Number of filtered test examples:", len(x_test))

print(y_train[0])

# plt.imshow(x_train[0, :, :, 0])
# plt.colorbar()

x_train_small = tf.image.resize(x_train, (4,4)).numpy()
x_test_small = tf.image.resize(x_test, (4,4)).numpy()

def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
       orig_x[tuple(x.flatten())] = x
       mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for flatten_x in mapping:
      x = orig_x[flatten_x]
      labels = mapping[flatten_x]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(next(iter(labels)))
      else:
          # Throw out images that match more than one label.
          pass

    num_uniq_class1 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    num_uniq_class2 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)

    print("Number of unique images:", len(mapping.values()))
    print(f"Number of unique {gt_class1}s: {num_uniq_class1}")
    print(f"Number of unique {gt_class2}s: {num_uniq_class2}")
    print(f"Number of unique contradicting labels (both {gt_class1} and {gt_class2}): {num_uniq_both}")
    print()
    print("Initial number of images: ", len(xs))
    print("Remaining non-contradicting unique images: ", len(new_x))

    return np.array(new_x), np.array(new_y)

x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)

# Encoding
phase_encoding = True

THRESHOLD = 0.5

if phase_encoding:
    x_train_bin = np.array(x_train_nocon, dtype=np.float32)
    x_test_bin = np.array(x_test_small, dtype=np.float32)
else:
    x_train_bin = np.array(x_train_nocon > THRESHOLD, dtype=np.float32)
    x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.float32)
    

_ = remove_contradicting(x_train_bin, y_train_nocon)

def convert_to_circuit(image,phase_encoding=False):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if phase_encoding:
            # inputs already in range [0,1]
            rot = cirq.XPowGate(exponent=value)
            circuit.append(rot(qubits[i]))
        else:
            if value:
                circuit.append(cirq.X(qubits[i]))
    return circuit


x_train_circ = [convert_to_circuit(x,phase_encoding) for x in x_train_bin]
x_test_circ = [convert_to_circuit(x,phase_encoding) for x in x_test_bin]

SVGCircuit(x_train_circ[0])

# Check first image
bin_img = x_train_bin[0,:,:,0]
indices = np.array(np.where(bin_img)).T
print(indices)


# QNN

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)

demo_builder = CircuitLayerBuilder(data_qubits = cirq.GridQubit.rect(4,1),
                                   readout=cirq.GridQubit(-1,-1))

circuit = cirq.Circuit()
demo_builder.add_layer(circuit, gate = cirq.XX, prefix='xx')
SVGCircuit(circuit)

def create_quantum_model():
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    # Then add layers (experiment by adding more).
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)

model_circuit, model_readout = create_quantum_model()


# Build the Keras model.
model = tf.keras.Sequential([
    # The input is the data-circuit, encoded as a tf.string
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    # The PQC layer returns the expected value of the readout gate, range [-1,1].
    tfq.layers.PQC(model_circuit, model_readout),
])

# Adapt Y labels for Hinge loss -> [-1,1]
y_train_hinge = 2.0*y_train_nocon-1.0
y_test_hinge = 2.0*y_test-1.0

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)

model.compile(
    loss=tf.keras.losses.Hinge(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[hinge_accuracy])

print(model.summary())

x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)


EPOCHS = 3
BATCH_SIZE = 32

NUM_EXAMPLES = len(x_train_tfcirc)

x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

qnn_history = model.fit(
      x_train_tfcirc_sub, y_train_hinge_sub,
      batch_size=32,
      epochs=EPOCHS,
      verbose=1,
      validation_data=(x_test_tfcirc, y_test_hinge))

qnn_results = model.evaluate(x_test_tfcirc, y_test)

# Classical model (few parameters)

def create_fair_classical_model():
    # A simple model based off LeNet from https://keras.io/examples/mnist_cnn/
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(4,4,1)))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model


model = create_fair_classical_model()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()

model.fit(x_train_bin,
          y_train_nocon,
          batch_size=128,
          epochs=20,
          verbose=2,
          validation_data=(x_test_bin, y_test))

fair_nn_results = model.evaluate(x_test_bin, y_test)

qnn_accuracy = qnn_results[1]
fair_nn_accuracy = fair_nn_results[1]

plt.clf()
sns.barplot(["Quantum", "Classical, fair"],
            [float(qnn_accuracy), float(fair_nn_accuracy)])
plt.savefig('mnist_comparison.jpg')

# Console expected output
"""
Number of original training examples: 60000
Number of original test examples: 10000
Number of filtered training examples: 12049
Number of filtered test examples: 1968
True
Number of unique images: 10387
Number of unique 3s:  4912
Number of unique 6s:  5426
Number of unique contradicting labels (both 3 and 6):  49

Initial number of images:  12049
Remaining non-contradicting unique images:  10338
Number of unique images: 193
Number of unique 3s:  80
Number of unique 6s:  69
Number of unique contradicting labels (both 3 and 6):  44

Initial number of images:  10338
Remaining non-contradicting unique images:  149
[[2 2]
 [3 1]]
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 pqc (PQC)                   (None, 1)                 32        
                                                                 
=================================================================
Total params: 32
Trainable params: 32
Non-trainable params: 0
_________________________________________________________________
None
Epoch 1/3
324/324 [==============================] - 306s 942ms/step - loss: 0.6890 - hinge_accuracy: 0.7824 - val_loss: 0.4140 - val_hinge_accuracy: 0.8931
Epoch 2/3
324/324 [==============================] - 312s 963ms/step - loss: 0.3877 - hinge_accuracy: 0.8764 - val_loss: 0.3675 - val_hinge_accuracy: 0.9012
Epoch 3/3
324/324 [==============================] - 296s 914ms/step - loss: 0.3639 - hinge_accuracy: 0.8857 - val_loss: 0.3513 - val_hinge_accuracy: 0.9042
62/62 [==============================] - 7s 108ms/step - loss: 0.3513 - hinge_accuracy: 0.9042
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 16)                0         
                                                                 
 dense (Dense)               (None, 2)                 34        
                                                                 
 dense_1 (Dense)             (None, 1)                 3         
                                                                 
=================================================================
Total params: 37
Trainable params: 37
Non-trainable params: 0
_________________________________________________________________
Epoch 1/20
81/81 - 0s - loss: 0.7226 - accuracy: 0.5285 - val_loss: 0.7010 - val_accuracy: 0.5422 - 449ms/epoch - 6ms/step
Epoch 2/20
81/81 - 0s - loss: 0.6639 - accuracy: 0.6442 - val_loss: 0.6573 - val_accuracy: 0.6194 - 78ms/epoch - 961us/step
Epoch 3/20
81/81 - 0s - loss: 0.6190 - accuracy: 0.6778 - val_loss: 0.6063 - val_accuracy: 0.6240 - 82ms/epoch - 1ms/step
Epoch 4/20
81/81 - 0s - loss: 0.5711 - accuracy: 0.7622 - val_loss: 0.5567 - val_accuracy: 0.8049 - 78ms/epoch - 957us/step
Epoch 5/20
81/81 - 0s - loss: 0.5279 - accuracy: 0.8280 - val_loss: 0.5137 - val_accuracy: 0.8100 - 80ms/epoch - 984us/step
Epoch 6/20
81/81 - 0s - loss: 0.4889 - accuracy: 0.8345 - val_loss: 0.4750 - val_accuracy: 0.8161 - 81ms/epoch - 1ms/step
Epoch 7/20
81/81 - 0s - loss: 0.4563 - accuracy: 0.8438 - val_loss: 0.4452 - val_accuracy: 0.8257 - 76ms/epoch - 937us/step
Epoch 8/20
81/81 - 0s - loss: 0.4310 - accuracy: 0.8500 - val_loss: 0.4221 - val_accuracy: 0.8267 - 78ms/epoch - 962us/step
Epoch 9/20
81/81 - 0s - loss: 0.4103 - accuracy: 0.8524 - val_loss: 0.4033 - val_accuracy: 0.8267 - 71ms/epoch - 880us/step
Epoch 10/20
81/81 - 0s - loss: 0.3926 - accuracy: 0.8531 - val_loss: 0.3872 - val_accuracy: 0.8267 - 76ms/epoch - 933us/step
Epoch 11/20
81/81 - 0s - loss: 0.3768 - accuracy: 0.8531 - val_loss: 0.3732 - val_accuracy: 0.8272 - 74ms/epoch - 911us/step
Epoch 12/20
81/81 - 0s - loss: 0.3626 - accuracy: 0.8590 - val_loss: 0.3598 - val_accuracy: 0.8313 - 76ms/epoch - 940us/step
Epoch 13/20
81/81 - 0s - loss: 0.3500 - accuracy: 0.8606 - val_loss: 0.3475 - val_accuracy: 0.8313 - 77ms/epoch - 956us/step
Epoch 14/20
81/81 - 0s - loss: 0.3386 - accuracy: 0.8608 - val_loss: 0.3366 - val_accuracy: 0.8313 - 78ms/epoch - 967us/step
Epoch 15/20
81/81 - 0s - loss: 0.3278 - accuracy: 0.8629 - val_loss: 0.3256 - val_accuracy: 0.8333 - 73ms/epoch - 902us/step
Epoch 16/20
81/81 - 0s - loss: 0.3179 - accuracy: 0.8637 - val_loss: 0.3159 - val_accuracy: 0.8333 - 77ms/epoch - 949us/step
Epoch 17/20
81/81 - 0s - loss: 0.3090 - accuracy: 0.8637 - val_loss: 0.3071 - val_accuracy: 0.8333 - 70ms/epoch - 865us/step
Epoch 18/20
81/81 - 0s - loss: 0.3011 - accuracy: 0.8687 - val_loss: 0.2995 - val_accuracy: 0.8720 - 74ms/epoch - 910us/step
Epoch 19/20
81/81 - 0s - loss: 0.2941 - accuracy: 0.8844 - val_loss: 0.2929 - val_accuracy: 0.8720 - 68ms/epoch - 842us/step
Epoch 20/20
81/81 - 0s - loss: 0.2879 - accuracy: 0.8845 - val_loss: 0.2866 - val_accuracy: 0.8720 - 74ms/epoch - 914us/step
62/62 [==============================] - 0s 576us/step - loss: 0.2866 - accuracy: 0.8720

"""