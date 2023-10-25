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
from torch import cat, clone
import torch.nn.functional as F
from qiskit_machine_learning.connectors import TorchConnector


#from qiskit_machine_learning.algorithms import VQC
#vqc = VQC(feature_map=ZZFeatureMap(num_qubits), ansatz=RealAmplitudes(num_qubits, reps=1), loss='cross_entropy', optimizer=L_BFGS_B(), quantum_instance=QasmSimulator())

class HybridQNN_Shallow(Module):
    def __init__(self,n_classes = 2, n_qubits = 2, n_channels = 3, n_filts_fc1 = 256, n_filts_fc2 = 64, qnn = None):
         
        # save for when computing head
        self.n_classes = n_classes

        super().__init__()
        self.conv1 = Conv2d(n_channels, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(n_filts_fc1, n_filts_fc2) # 256
        self.fc2 = Linear(n_filts_fc2,n_qubits)  # 2-dimensional input to QNN
        if qnn is not None:
            self.qnn = TorchConnector(qnn)  # Apply torch connector, weights chosen
        else:
            self.qnn = None
        # uniformly at random from interval [-1,1].
        if self.n_classes == 2:
            self.fc3 = Linear(1, 1) # 1-dimensional output from QNN
        else:
            self.fc3 = Linear(1, n_classes) 
    def forward(self, x):
        x = F.relu(self.conv1(x)) # -4
        x = F.max_pool2d(x, 2) # / 2
        x = F.relu(self.conv2(x)) # -4
        x = F.max_pool2d(x, 2) # / 2
        x = self.dropout(x) #  
        x = x.view(x.shape[0], -1) # **2 then *16
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.qnn is not None:
            x = self.qnn(x)  # apply QNN
        x = self.fc3(x)
        if self.n_classes == 2:
            return cat((x, 1 - x), -1)
        else:
            return x

from qiskit import Aer, QuantumCircuit, BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import AerPauliExpectation
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector

class HybridQNN(Module):
    def __init__(self,backbone="HybridQNN_Shallow",**kwargs):
        ## Load Network
        super().__init__()
        
        # save main arguments from kwargs
        if 'n_channels' in kwargs:
            self.n_channels = kwargs['n_channels']
        else:
            self.n_channels = None
            
        if 'n_classes' in kwargs:
            self.n_classes = kwargs['n_classes']
            self.n_head_classes = self.n_classes
        else:
            self.n_classes = 2
            self.n_head_classes = 1
            
        if 'n_qubits' in kwargs:
            self.n_qubits = kwargs['n_qubits']
        else:
            self.n_qubits = 2  
        if 'n_features' in kwargs:
            self.n_features = kwargs['n_features']
        else:
            self.n_features = self.n_qubits
        
        # remove extra kwargs
        if "n_channels" in kwargs:
            del kwargs["n_channels"]
        if "n_classes" in kwargs:
            del kwargs["n_classes"]
        if "n_qubits" in kwargs:
            del kwargs["n_qubits"]
        if "n_features" in kwargs:
            del kwargs["n_features"]
        
        # build network
        
        if backbone in globals(): # globals() are previously defined in this python file
            self.network = globals()[backbone](**kwargs)
        if backbone not in globals():
            import_str = f'from torchvision.models import {backbone}'
            exec(import_str)
            self.network = locals()[backbone](**kwargs) # locals() are defined just in this function
        #print(self.network)
                
        
        # DEFINE QNN Network
        self.set_simulator("aer_simulator_statevector", **kwargs)
        #print(self.qnn)
        # inject qnn right before head
        
        # Redefine channels
        if hasattr(self.network,'conv1'):
            self.network.conv1 = Conv2d(in_channels=self.n_channels, out_channels=self.network.conv1.out_channels, kernel_size=self.network.conv1.kernel_size, stride=self.network.conv1.stride, padding=self.network.conv1.padding,bias=self.network.conv1.bias)
        
        # Redefine head
        if hasattr(self.network,'fc'): # HEAD = fc
            if type(self.network.fc) == Linear:
                in_features = int(self.network.fc.in_features)
                self.network.fc = Linear(1, self.n_head_classes) 
                #replacement here
                self.network.fc = Sequential(
                        Linear(in_features,self.n_features),TorchConnector(self.qnn), self.network.fc
                    )
            elif type(self.network.fc) == Sequential:
                list_head = [n for n, _ in self.network.fc.named_children()]
                for layer in reversed(list_head):
                    if type(self.network.fc[int(layer)]) == Linear:
                        in_features = int(self.network.fc[int(layer)].in_features)
                        self.network.fc[int(layer)] = Linear(1, self.n_head_classes)
                        #replacement here
                        self.network.fc[int(layer)] = Sequential(
                                Linear(in_features,self.n_features),TorchConnector(self.qnn), self.network.fc[int(layer)]
                            )
                        break
        elif hasattr(self.network,'classifier'): # HEAD = classifier
            if type(self.network.classifier) == Linear:
                in_features = int(self.network.classifier.in_features)
                self.network.classifier = Linear(1, self.n_head_classes) 
                #replacement here
                self.network.classifier = Sequential(
                        Linear(in_features,self.n_features),TorchConnector(self.qnn), self.network.classifier
                    )
            elif type(self.network.classifier) == Sequential:
                list_head = [n for n, _ in self.network.classifier.named_children()]
                for layer in reversed(list_head):
                    if type(self.network.classifier[int(layer)]) == Linear:
                        in_features = int(self.network.classifier[int(layer)].in_features)
                        self.network.classifier[int(layer)] = Linear(1, self.n_head_classes)
                        #replacement here
                        self.network.classifier[int(layer)] = Sequential(
                                Linear(in_features,self.n_features),TorchConnector(self.qnn), self.network.classifier[int(layer)]
                            )
                        break
        else: # HEAD = any name
            list_head = [n for n, _ in self.network.named_children()]
            for layer in reversed(list_head):
                if type(self.network[layer]) == Linear:
                    in_features = int(self.network[layer].in_features)
                    self.network[layer] = Linear(1, self.n_head_classes) 
                    #replacement here
                    self.network[layer] = Sequential(
                            Linear(in_features,self.n_features),TorchConnector(self.qnn), self.network[layer]
                        )
                    break
                elif type(self.network[layer]) == Sequential:
                    list_subhead = [n for n, _ in self.network[layer].named_children()]
                    for sublayer in reversed(list_subhead):
                        if type(self.network[layer][sublayer]) == Linear:
                            in_features = int(self.network[layer][sublayer].in_features)
                            self.network[layer][sublayer] = Linear(1, self.n_head_classes) 
                            #replacement here
                            self.network[layer][sublayer] = Sequential(
                                    Linear(in_features,self.n_features),TorchConnector(self.qnn), self.network[layer][sublayer]
                                )
                            break
        # print(self.network)
        
    def forward(self, x):
        x = self.network(x)
        if self.n_classes == 2:
            return cat((x, 1 - x), -1)
        else:
            return x
        
    def set_simulator(self, simulator="aer_simulator_statevector", **kwargs ):
        # declare quantum instance
        self.qi = QuantumInstance(Aer.get_backend(simulator))
        # Define QNN
        self.feature_map = ZZFeatureMap(self.n_features)
        self.ansatz = RealAmplitudes(self.n_qubits, reps=1)
        # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
        self.qnn = TwoLayerQNN(
            self.n_qubits, self.feature_map, self.ansatz, input_gradients=True, exp_val=AerPauliExpectation(), quantum_instance=self.qi
        )
        print(self.qnn.operator)
        self.qnn.circuit.draw(output="mpl",filename=f"plots/qnn{self.n_qubits}_{self.n_classes}classes.png")
        #from qiskit.quantum_info import Statevector
        #from qiskit.visualization import plot_bloch_multivector
        #state = Statevector.from_instruction(self.qnn.circuit)
        #plot_bloch_multivector(state)
   
        
        