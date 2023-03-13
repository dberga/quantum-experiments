import random

from qiskit import transpile
from qiskit.circuit.random import random_circuit


# Class to serialize between python / JSON
class JSONSerializer:
    def __init__(self, **kwargs):
        self.params = kwargs

    def to_json(self):
        return json.dumps(self.params)

    @classmethod
    def from_json(cls, json_str):
        return cls(**json.loads(json_str))
    
def prepare_circuits(backend,num_qubits,depth):
    """Generate a random circuit.

    Args:
        backend: Backend used for transpilation.

    Returns:
        Generated circuit.
    """
    circuit = random_circuit(
        num_qubits=num_qubits, depth=depth, measure=True, seed=random.randint(0, 1000)
    )
    return transpile(circuit, backend)


def main(backend, user_messenger, **kwargs):
    """Main entry point of the program.

    Args:
        backend: Backend to submit the circuits to.
        user_messenger: Used to communicate with the program consumer.
        kwargs: User inputs.
    """
    iterations = kwargs.pop("iterations", 5)

    
    num_qubits = kwargs.pop('num_qubits')
    depth = kwargs.pop('depth')
    for it in range(iterations):
        qc = prepare_circuits(backend, num_qubits,depth)
        result = backend.run(qc).result()
        user_messenger.publish({"iteration": it, "counts": result.get_counts()})

    return "All done!"