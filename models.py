from quimb.tensor import Circuit
from quimb import pauli

from itertools import combinations

def local_gates(N: int, max_weight: int = 2):
    """
    Create a list of lists of all possible combinations of integers with a given weight

    Parameters
    ----------
    N: int
        Number of integers to combine
    max_weight: int
        Maximum weight of each list

    """
    return [
        list(gate) 
        for w in range(1, max_weight + 1) 
        for gate in combinations(range(N), w)
    ]


def RStringZ(circ: Circuit, qubits: list, theta: float):
    """
    Apply in a Quimb circuit a multi-qubit Z-rotation (exp(-i * theta/2 * Z1 ⊗ Z2 ⊗ ... ⊗ Zn)) 
    to the circuit using CNOT strings and an RZ gate.

    Parameters
    ----------
    circ : quimb.tensor.Circuit
        The quimb circuit object to which the gate is applied.
    qubits : list[int]
        The indices of the qubits involved in the interaction.
    theta : float
        The rotation angle.
    """
    if not qubits:
        raise ValueError("La lista dei qubits non può essere vuota.")
    
    if len(qubits) == 1:
        circ.apply_gate('RZ', theta, qubits[0], parametrize=True)
        return
    
    if len(qubits) == 2:
        circ.apply_gate('RZZ', theta, *qubits, parametrize=True)
        return
    

    for i in range(len(qubits) - 1):
        circ.apply_gate('CX', qubits[i], qubits[i+1])
    circ.apply_gate('RZ', theta, qubits[-1], parametrize=True)
    for i in reversed(range(len(qubits) - 1)):
        circ.apply_gate('CX', qubits[i], qubits[i+1])


class IQPTensorNetwork:
    """
    Represent an Instantaneous Quantum Polynomial (IQP) circuit using quimb.

    An IQP circuit consists of a layer of Hadamard gates, followed by a series 
    of commuting Z-diagonal gates (rotations based on interactions), and a 
    final layer of Hadamard gates.

    Parameters
    ----------
    nqubits : int
        The number of qubits in the circuit.
    interactions : list of lists/tuples
        A list where each element contains the indices of qubits involved in a 
        rotation gate.
    """
    def __init__(self, nqubits, interactions):

        self.nqubits = nqubits
        self.interactions = interactions

    def build_circuit(self, parameters):
        """
        Build the IQP circuit as a quimb skip-gate Circuit object.

        Parameters
        ----------
        params : array-like
            The rotation angles (thetas) for each interaction. Should have the 
            same length as `self.interactions`.

        Returns
        -------
        qtn.Circuit
            The constructed quimb quantum circuit.
        """
        circ = Circuit(self.nqubits)

        for i in range(self.nqubits):
            circ.apply_gate('H', i)
        
        for theta, gen in zip(parameters, self.interactions):
            RStringZ(circ, gen, theta)

        for i in range(self.nqubits):
            circ.apply_gate('H', i)

        return circ
    