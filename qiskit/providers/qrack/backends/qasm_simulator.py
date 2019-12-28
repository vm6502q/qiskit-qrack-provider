# This code is based on and adapted from https://github.com/Qiskit/qiskit-qcgpu-provider/blob/master/qiskit_qcgpu_provider/qasm_simulator.py
#
# Adapted by Daniel Strano. Many thanks to Adam Kelley for pioneering a third-party Qiskit provider.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name


import uuid
import time
import numpy as np
from collections import Counter
import logging
import os
from math import log2
from qiskit.util import local_hardware_info
from qiskit.providers.models import BackendConfiguration
from .qrack_controller_wrapper import qrack_controller_factory
from ..qrackerror import QrackError
from ..version import __version__

from ..qrackjob import QrackJob
from ..qrackerror import QrackError

from qiskit.providers import BaseBackend
from qiskit.result import Result

logger = logging.getLogger(__name__)


class QasmSimulator(BaseBackend):
    """Contains an OpenCL based backend"""

    DEFAULT_CONFIGURATION = {
        'backend_name': 'qasm_simulator',
        'backend_version': __version__,
        'n_qubits': 64,
        'url': 'https://github.com/vm6502q/qiskit-aer',
        'simulator': True,
        'local': True,
        'conditional': False,
        'open_pulse': False,
        'memory': True,
        'max_shots': 65536,
        'description': 'An OpenCL based qasm simulator',
        'coupling_map': None,
        'basis_gates': [
            'u1', 'u2', 'u3', 'cx', 'cz', 'ch', 'id', 'x', 'y', 'z', 'h', 'rx', 'ry',
            'rz', 's', 'sdg', 't', 'tdg', 'swap', 'ccx', 'cu1', 'cu2', 'cu3', 'cswap',
            'mcx', 'mcy', 'mcz', 'mcu1', 'mcu2', 'mcu3', 'mcswap'
        ]
        'gates': [{
            'name': 'u1',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'Single-qubit gate [[1, 0], [0, exp(1j*lam)]]',
            'qasm_def': 'gate u1(lam) q { U(0,0,lam) q; }'
        }, {
            'name': 'u2',
            'parameters': ['phi', 'lam'],
            'conditional': True,
            'description':
            'Single-qubit gate [[1, -exp(1j*lam)], [exp(1j*phi), exp(1j*(phi+lam))]]/sqrt(2)',
            'qasm_def': 'gate u2(phi,lam) q { U(pi/2,phi,lam) q; }'
        }, {
            'name':
            'u3',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional':
            True,
            'description':
            'Single-qubit gate with three rotation angles',
            'qasm_def':
            'gate u3(theta,phi,lam) q { U(theta,phi,lam) q; }'
        }, {
            'name': 'cx',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit Controlled-NOT gate',
            'qasm_def': 'gate cx c,t { CX c,t; }'
        }, {
            'name': 'cz',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit Controlled-Z gate',
            'qasm_def': 'gate cz a,b { h b; cx a,b; h b; }'
        }, {
            'name': 'ch',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit Controlled-H gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'id',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit identity gate',
            'qasm_def': 'gate id a { U(0,0,0) a; }'
        }, {
            'name': 'x',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-X gate',
            'qasm_def': 'gate x a { U(pi,0,pi) a; }'
        }, {
            'name': 'y',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-Y gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'z',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-Z gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'h',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Hadamard gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'rx',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-X axis rotation gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'ry',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-Y axis rotation gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'rz',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-Z axis rotation gate',
            'qasm_def': 'TODO'
        }, {
            'name': 's',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit phase gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'sdg',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit adjoint phase gate',
            'qasm_def': 'TODO'
        }, {
            'name': 't',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit T gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'tdg',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit adjoint T gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'swap',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit SWAP gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'ccx',
            'parameters': [],
            'conditional': True,
            'description': 'Three-qubit Toffoli gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'cswap',
            'parameters': [],
            'conditional': True,
            'description': 'Three-qubit Fredkin (controlled-SWAP) gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'cu1',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'Two-qubit Controlled-u1 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'cu2',
            'parameters': ['phi', 'lam'],
            'conditional': True,
            'description': 'Two-qubit Controlled-u2 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'cu3',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional': True,
            'description': 'Two-qubit Controlled-u3 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcx',
            'parameters': [],
            'conditional': True,
            'description': 'N-qubit multi-controlled-X gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcy',
            'parameters': [],
            'conditional': True,
            'description': 'N-qubit multi-controlled-Y gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcz',
            'parameters': [],
            'conditional': True,
            'description': 'N-qubit multi-controlled-Z gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcu1',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'N-qubit multi-controlled-u1 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcu2',
            'parameters': ['phi', 'lam'],
            'conditional': True,
            'description': 'N-qubit multi-controlled-u2 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcu3',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional': True,
            'description': 'N-qubit multi-controlled-u3 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcswap',
            'parameters': [],
            'conditional': True,
            'description': 'N-qubit multi-controlled-SWAP gate',
            'qasm_def': 'TODO'
        }]

    def __init__(self, configuration=None, provider=None):
        configuration = configuration or BackendConfiguration.from_dict(
            self.DEFAULT_CONFIGURATION)
        super().__init__(configuration=configuration, provider=provider)

        self._configuration = configuration
        self._number_of_qubits = None
        self._number_of_cbits = None
        self._statevector = None
        self._results = {}
        self._shots = {}
        self._local_random = np.random.RandomState()
        self._sample_measure = False
        self._chop_threshold = 15  # chop to 10^-15

    #@profile
    def run(self, qobj):
        """Run qobj asynchronously.
        Args:
            qobj (Qobj): payload of the experiment
        Returns:
            QrackJob: derived from BaseJob
        """

        job_id = str(uuid.uuid4())
        job = QrackJob(self, job_id, self._run_job(job_id, qobj), qobj)
        return job
    #@profile
    def _run_job(self, job_id, qobj):
        """Run experiments in qobj
        Args:
            job_id (str): unique id for the job.
            qobj (Qobj): job description
        Returns:
            Result: Result object
        """
        self._shots = qobj.config.shots
        self._memory = qobj.config.memory
        self._qobj_config = qobj.config
        results = []

        start = time.time()
        for experiment in qobj.experiments:
            results.append(self.run_experiment(experiment))
        end = time.time()

        result = {
            'backend_name': self.name(),
            'backend_version': self._configuration.backend_version,
            'qobj_id': qobj.qobj_id,
            'job_id': job_id,
            'results': results,
            'status': 'COMPLETED',
            'success': True,
            'time_taken': (end - start),
            'header': qobj.header.to_dict()
        }

        return Result.from_dict(result)

    #@profile
    def run_experiment(self, experiment):
        """Run an experiment (circuit) and return a single experiment result.
        Args:
            experiment (QobjExperiment): experiment from qobj experiments list
        Returns:
            dict: A dictionary of results.
            dict: A result dictionary
        Raises:
            QrackSimulatorError: If the number of qubits is too large, or another
                error occurs during execution.
        """
        self._number_of_qubits = experiment.header.n_qubits
        self._number_of_cbits = experiment.header.memory_slots
        self._classical_state = 0
        self._statevector = 0

        if hasattr(experiment.config, 'seed'):
            seed = experiment.config.seed
        elif hasattr(self._qobj_config, 'seed'):
            seed = self._qobj_config.seed
        else:
            # For compatibility on Windows force dyte to be int32
            # and set the maximum value to be (2 ** 31) - 1
            seed = np.random.randint(2147483647, dtype='int32')
        self._local_random.seed(seed)

        self._can_sample(experiment)

        if not self._sample_measure:
            raise QrackSimulatorError('Measurements are only supported at the end')

        experiment = experiment.to_dict()

        samples = []

        start = time.time()

        try:
            sim = qrack_controller_factory()
            sim.initialize_qreg(self._number_of_qubits)
        except OverflowError:
            raise QrackSimulatorError('too many qubits')

        for operation in experiment['instructions']:
            name = operation['name']

            if name == 'u1':
                sim.u1(operation['qubits'][0], operation['params'])
            elif name == 'u2':
                sim.u2(operation['qubits'][0], operation['params'])
            elif name == 'u3':
                sim.u(operation['qubits'][0], operation['params'])
            elif name == 'cx':
                sim.cx(operation['qubits'], 1)
            elif name == 'cz':
                sim.cz(operation['qubits'], 1)
            elif name == 'ch':
                sim.ch(operation['qubits'], 1)
            elif name == 'id':
                logger.info('Identity gates are ignored.')
            elif name == 'x':
                sim.x(operation['qubits'][0])
            elif name == 'y':
                sim.y(operation['qubits'][0])
            elif name == 'z':
                sim.z(operation['qubits'][0])
            elif name == 'h':
                sim.h(operation['qubits'][0])
            elif name == 'rx':
                sim.rx(operation['qubits'][0], operation['params'])
            elif name == 'ry':
                sim.ry(operation['qubits'][0], operation['params'])
            elif name == 'rz':
                sim.rz(operation['qubits'][0], operation['params'])
            elif name == 's':
                sim.s(operation['qubits'][0])
            elif name == 'sdg':
                sim.sdg(operation['qubits'][0])
            elif name == 't':
                sim.t(operation['qubits'][0])
            elif name == 'tdg':
                sim.tdg(operation['qubits'][0])
            elif name == 'swap':
                sim.swap(operation['qubits'][0], operation['qubits'][1])
            elif name == 'ccx':
                sim.cx(operation['qubits'], 2)
            elif name == 'cu1':
                sim.cu1(operation['qubits'], 1, operation['params'])
            elif name == 'cu2':
                sim.cu2(operation['qubits'], 1, operation['params'])
            elif name == 'cu3':
                sim.cu(operation['qubits'], 1, operation['params'])
            elif name == 'cswap':
                sim.cswap(operation['qubits'], 1)
            elif name == 'mcx':
                sim.cx(operation['qubits'], len(operation['qubits']) - 1)
            elif name == 'mcy':
                sim.cy(operation['qubits'], len(operation['qubits']) - 1)
            elif name == 'mcz':
                sim.cz(operation['qubits'], len(operation['qubits']) - 1)
            elif name == 'cu1':
                sim.cu1(operation['qubits'], len(operation['qubits']) - 1, operation['params'])
            elif name == 'cu2':
                sim.cu2(operation['qubits'], len(operation['qubits']) - 1, operation['params'])
            elif name == 'cu3':
                sim.cu(operation['qubits'], len(operation['qubits']) - 1, operation['params'])
            elif name == 'mcswap':
                sim.cswap(operation['qubits'], len(operation['qubits']) - 2)
            elif name == 'reset':
                sim.reset(operation['qubits'][0])
            elif name == 'measure':
                samples.append((operation['qubits'][0], operation['memory'][0]))
            elif name == 'barrier':
                logger.info('Barrier gates are ignored.')

        if self._number_of_cbits > 0:
            memory = self._add_sample_measure(samples, sim, self._shots)
        else:
            memory = []

        end = time.time()

        # amps = sim.amplitudes().round(self._chop_threshold)
        # amps = np.stack((amps.real, amps.imag), axis=-1)

        data = {'counts': dict(Counter(memory))}

        if self._memory:
            data['memory'] = memory

        return {
            'name': experiment['header']['name'],
            'shots': self._shots,
            'data': data,
            'seed': seed,
            'status': 'DONE',
            'success': True,
            'time_taken': (end - start),
            'header': experiment['header']
        }

    #@profile
    def _add_sample_measure(self, measure_params, sim, num_samples):
        """Generate memory samples from current statevector.
        Taken almost straight from the terra source code.
        Args:
            measure_params (list): List of (qubit, clbit) values for
                                   measure instructions to sample.
            num_samples (int): The number of memory samples to generate.
        Returns:
            list: A list of memory values in hex format.
        """
        memory = []
        # If we only want one sample, it's faster for the backend to do it,
        # without passing back the probabilities.
        if num_samples == 1:
            sample = sim.measure_all()
            classical_state = self._classical_state
            for qubit, cbit in measure_params:
                qubit_outcome = int((sample & (1 << qubit)) >> qubit)
                bit = 1 << cbit
                classical_state = (classical_state & (~bit)) | (qubit_outcome << cbit)
            value = bin(classical_state)[2:]
            memory.append(hex(int(value, 2)))
            return memory

        probabilities = np.reshape(sim.probabilities(), self._number_of_qubits * [2])

        # Get unique qubits that are actually measured
        measured_qubits = list(set([qubit for qubit, clbit in measure_params]))
        num_measured = len(measured_qubits)


        # Axis for numpy.sum to compute probabilities
        axis = list(range(self._number_of_qubits))

        for qubit in reversed(measured_qubits):
            # Remove from largest qubit to smallest so list position is correct
            # with respect to position from end of the list
            axis.remove(self._number_of_qubits - 1 - qubit)
        
        
        probabilities = np.reshape(np.sum(probabilities,
                                          axis=tuple(axis)),
                                   2 ** num_measured)
        # Generate samples on measured qubits
        samples = self._local_random.choice(range(2 ** num_measured),
                                            num_samples, p=probabilities)
        # Convert to bit-strings
        for sample in samples:
            classical_state = self._classical_state
            for qubit, cbit in measure_params:
                qubit_outcome = int((sample & (1 << qubit)) >> qubit)
                bit = 1 << cbit
                classical_state = (classical_state & (~bit)) | (qubit_outcome << cbit)
            value = bin(classical_state)[2:]
            memory.append(hex(int(value, 2)))
        return memory

    #@profile
    def _can_sample(self, experiment):
        """Determine if sampling can be used for an experiment
        Args:
            experiment (QobjExperiment): a qobj experiment
        """
        measure_flags = {}
        if hasattr(experiment.config, 'allows_measure_sampling'):
            self._sample_measure = experiment.config.allows_measure_sampling
        else:

            for instruction in experiment.instructions:
                if instruction.name == "reset":
                    measure_flags[instruction.qubits[0]] = False
                    self._sample_measure = False
                    return

                if measure_flags.get(instruction.qubits[0], False):
                    if instruction.name not in ["measure", "barrier", "id", "u0"]:
                        for qubit in instruction.qubits:
                            measure_flags[qubit] = False
                            return
                elif instruction.name == "measure":
                     for qubit in instruction.qubits:
                            measure_flags[qubit] = True
        
        self._sample_measure = True

        for key, value in measure_flags.items():
            if value == False:
                self._sample_measure = False
                return

    @staticmethod
    def name():
        return 'qasm_simulator'
