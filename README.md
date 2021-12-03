# qiskit-qrack-provider

This repository contains a Qrack provider for Qiskit. You must [install PyQrack](https://pypi.org/project/pyqrack/) to use it.

The underlying Qrack simulator is a high-performance, GPU-accelarated, noiseless simulator, by design. This repository provides the Qrack `QasmSimulator`.

This provider is based on and adapted from work by the IBM Qiskit Team and QCGPU's creator, Adam Kelly. Attribution is noted in content files, where appropriate. Original contributions and adaptations were made by Daniel Strano of the VM6502Q/Qrack Team.

To use, in Qiskit:
```python
from qiskit.providers.qrack import Qrack

backend = Qrack.get_backend('qasm_simulator')
```

For example, for use with [unitaryfund/mitiq](https://github.com/unitaryfund/mitiq), creating a (noiseless) `executor` can be as simple as follows:
```python
from qiskit.providers.qrack import Qrack

def executor(circuit, shots=1000):
    """Executes the input circuit and returns the noisy expectation value <A>, where A=|0><0|.
    """
    # Use the Qrack QasmSimulator backend, (but it's specifically noiseless)
    ideal_backend = Qrack.get_backend('qasm_simulator')

    # Append measurements
    circuit_to_run = circuit.copy()
    circuit_to_run.measure_all()

    # Run and get counts
    print(f"Executing circuit with {len(circuit)} gates using {shots} shots.")
    job = ideal_backend.run(circuit_to_run, shots=shots)
    counts = job.result().get_counts()

    # Compute expectation value of the observable A=|0><0|
    return counts["0"] / shots
```

Generally, you will need to adapt the above `executor` snippet to your particular purpose.

(Happy Qracking! You rock!)
