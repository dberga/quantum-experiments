# Use QuantumNas to train, search, train from scratch, prune, and eval step by step

Here we show an example to solve an MNIST-0123 task.

The target device is IBMQ_Lima, and circuit design space is U3+CU3. 

<mark>Here is a colab [link](https://colab.research.google.com/drive/1eWTEEkLJREanJ7GKvkjD4qrbxONQdrfz?usp=sharing) that runs the example in folder `artifact/example3/quantumnas/`.</mark>

If you are not using colab, you can follow the instructions below to install and run on your own computers.

## Installation
```bash
git clone https://github.com/Hanrui-Wang/pytorch-quantum.git
cd pytorch-quantum
pip install --editable .
pip install pathos
pip install tensorflow_model_optimization
export PYTHONPATH=.
```

Now your qiskit version should be 0.32.1. Modify the part after line 346 of `lib/python3.8/site-packages/qiskit/providers/aer/backends/aerbackend.py` to this:
```python
elif parameter_binds:
            # Handle parameter binding
            # parameterizations = self._convert_binds(circuits, parameter_binds)
            # assemble_binds = []
            # assemble_binds.append({param: 1 for bind in parameter_binds for param in bind})

            qobj = assemble(circuits, self, parameter_binds=parameter_binds)
```

Next entor the following code into the python interpreter to store a qiskit token to your local file. You can replace it with your own token from your IBMQ account.
```python
from qiskit import IBMQ
IBMQ.save_account('0238b0afc0dc515fe7987b02706791d1719cb89b68befedc125eded0607e6e9e9f26d3eed482f66fdc45fdfceca3aab2edb9519d96b39e9c78040194b86e7858', overwrite=True)
```

## Train a super circuit
```bash
bash artifact/example3/quantumnas/1_train_supercircuit.sh
```

## Evolutionary search
```bash
bash artifact/example3/quantumnas/2_search.sh
```

## Train the searched sub circuit from scratch
```bash
bash artifact/example3/quantumnas/3_train_subcircuit.sh
```

## Iterative pruning
```bash
bash artifact/example3/quantumnas/4_prune.sh
```

## Evaluate on real QC
```bash
bash artifact/example3/quantumnas/5_eval.sh
```

# Train and evaluate a human designed circuit


## Train
```bash
bash artifact/example3/human/1_train.sh
```

## Eval
```bash
bash artifact/example3/human/2_eval.sh
```