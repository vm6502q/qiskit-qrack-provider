# This code is based on and adapted from https://github.com/Qiskit/qiskit-qcgpu-provider/blob/master/qiskit_qcgpu_provider/statevector_simulator.py
#
# Adapted by Daniel Strano
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Qrack Statevector simulator backend.
"""

import uuid
import time
import numpy as np
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

# Logger
logger = logging.getLogger(__name__)


class StatevectorSimulator(BaseBackend):
    """Contains an OpenCL based backend

    **Backend options**

    The following backend options may be used with in the
    ``backend_options`` kwarg for :meth:`StatevectorSimulator.run` or
    ``qiskit.execute``:

    * ``"zero_threshold"`` (double): Sets the threshold for truncating
      small values to zero in the result data (Default: 1e-10).

    * ``"schmidt_decompose"`` (bool): If true, enable "QUnit" layer of
      Qrack, including Schmidt decomposition optimizations

    * ``"gate_fusion"`` (bool): If true, enable "QFusion" layer of
      Qrack, which attempts compose subsequent gates (at polynomial
      cost) before applying them (at exponential cost)

    * ``"opencl"`` (bool): If true, use the OpenCL engine of Qrack
      ("QEngineOCL") as the base "Schroedinger method" simulator.
      If OpenCL is not available, simulation will fall back to CPU.

    * ``"opencl_device_id"`` (int): (If OpenCL is enabled,) choose
      the OpenCl device to simulate on, (indexed by order of device
      discovery on OpenCL load/compilation). "-1" indicates to use
      the Qrack default device, (the last discovered, which tends to
      be a non-CPU accelerator, on common personal hardware systems.)
    """

    DEFAULT_CONFIGURATION = {
        'backend_name': 'statevector_simulator',
        'backend_version': __version__,
        'n_qubits': 64,
        'url': 'https://github.com/vm6502q/qiskit-aer',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': 1,
        'description': 'A Qrack-based, GPU-accelerated, C++ statevector simulator for qobj files',
        'coupling_map': None,
        'zero_threshold': -999.0,
        'schmidt_decompose': True,
        'gate_fusion': True,
        'opencl': True,
        'opencl_device_id': -1,
        'basis_gates': [
            'u1', 'u2', 'u3', 'cx', 'cz', 'ch', 'id', 'x', 'y', 'z', 'h', 'rx', 'ry',
            'rz', 's', 'sdg', 't', 'tdg', 'swap', 'ccx', 'initialize', 'cu1', 'cu2',
            'cu3', 'cswap', 'mcx', 'mcy', 'mcz', 'mcu1', 'mcu2', 'mcu3', 'mcswap',
            'multiplexer'
        ],
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
            'name': 'initialize',
            'parameters': ['vector'],
            'conditional': False,
            'description': 'N-qubit state initialize. '
                           'Resets qubits then sets statevector to the parameter vector.',
            'qasm_def': 'initialize(vector) q1, q2,...'
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
        }, {
            'name': 'multiplexer',
            'parameters': ['mat1', 'mat2', '...'],
            'conditional': True,
            'description': 'N-qubit multi-plexer gate. '
                           'The input parameters are the gates for each value.'
                           'WARNING: Qrack currently only supports single-qubit-target multiplexer gates',
            'qasm_def': 'TODO'
        }],
        # Location where we put external libraries that will be loaded at runtime
        # by the simulator extension
        'library_dir': os.path.dirname(__file__)
    }

    def __init__(self, configuration=None, provider=None):
        configuration = configuration or BackendConfiguration.from_dict(
            self.DEFAULT_CONFIGURATION)
        super().__init__(configuration=configuration, provider=provider)

        self._configuration = configuration
        self._number_of_qubits = None
        self._statevector = None
        self._results = {}
        self._chop_threshold = None # chop to 10^-8

    def run(self, qobj, backend_options={}):
        """
        Run qobj asynchronously.
        Args:
            qobj (Qobj): payload of the experiment
            backend_options: (ignored)
        Returns:
            QCGPUJob: derived from BaseJob
        """

        job_id = str(uuid.uuid4())
        job = QrackJob(self, job_id, self._run_job(job_id, qobj), qobj)
        return job

    def _run_job(self, job_id, qobj):
        """
        Run experiments in qobj
        Args:
            job_id (str): unique id for the job.
            qobj (Qobj): job description
        Returns:
            Result: Result object
        """
        self._validate(qobj)
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

        return Result.from_dict(result) # This can be sped up

    def run_experiment(self, experiment):
        """Run an experiment (circuit) and return a single experiment result.
        Args:
            experiment (QobjExperiment): experiment from qobj experiments list
        Returns:
            dict: A dictionary of results.
            dict: A result dictionary
        Raises:
            QCGPUSimulatorError: If the number of qubits is too large, or another
                error occurs during execution.
        """
        self._number_of_qubits = experiment.header.n_qubits
        self._statevector = 0
        experiment = experiment.to_dict()

        sample_qubits = []
        sample_clbits = []

        start = time.time()

        try:
            sim = qrack_controller_factory()
            sim.initialize_qreg(self._configuration.opencl,
                                self._configuration.gate_fusion,
                                self._configuration.schmidt_decompose,
                                self._number_of_qubits,
                                self._configuration.opencl_device_id,
                                self._configuration.zero_threshold)
        except OverflowError:
            raise QrackError('too many qubits')

        for operation in experiment['instructions']:
            name = operation['name']

            if name == 'u1':
                sim.u1(operation['qubits'], operation['params'])
            elif name == 'u2':
                sim.u2(operation['qubits'], operation['params'])
            elif name == 'u3':
                sim.u(operation['qubits'], operation['params'])
            elif name == 'cx':
                sim.cx(operation['qubits'])
            elif name == 'cz':
                sim.cz(operation['qubits'])
            elif name == 'ch':
                sim.ch(operation['qubits'])
            elif name == 'id':
                logger.info('Identity gates are ignored.')
            elif name == 'x':
                sim.x(operation['qubits'])
            elif name == 'y':
                sim.y(operation['qubits'])
            elif name == 'z':
                sim.z(operation['qubits'])
            elif name == 'h':
                sim.h(operation['qubits'])
            elif name == 'rx':
                sim.rx(operation['qubits'], operation['params'])
            elif name == 'ry':
                sim.ry(operation['qubits'], operation['params'])
            elif name == 'rz':
                sim.rz(operation['qubits'], operation['params'])
            elif name == 's':
                sim.s(operation['qubits'])
            elif name == 'sdg':
                sim.sdg(operation['qubits'])
            elif name == 't':
                sim.t(operation['qubits'])
            elif name == 'tdg':
                sim.tdg(operation['qubits'])
            elif name == 'swap':
                sim.swap(operation['qubits'][0], operation['qubits'][1])
            elif name == 'ccx':
                sim.cx(operation['qubits'])
            elif name == 'cu1':
                sim.cu1(operation['qubits'], operation['params'])
            elif name == 'cu2':
                sim.cu2(operation['qubits'], operation['params'])
            elif name == 'cu3':
                sim.cu(operation['qubits'], operation['params'])
            elif name == 'cswap':
                sim.cswap(operation['qubits'])
            elif name == 'mcx':
                sim.cx(operation['qubits'])
            elif name == 'mcy':
                sim.cy(operation['qubits'])
            elif name == 'mcz':
                sim.cz(operation['qubits'])
            elif name == 'initialize':
                print("initialize")
                sim.initialize(operation['qubits'], operation['params'])
            elif name == 'cu1':
                sim.cu1(operation['qubits'], operation['params'])
            elif name == 'cu2':
                sim.cu2(operation['qubits'], operation['params'])
            elif name == 'cu3':
                sim.cu(operation['qubits'], operation['params'])
            elif name == 'mcswap':
                sim.cswap(operation['qubits'])
            elif name == 'multiplexer':
                if len(operation['params'][0]) != 2: #matrix row count, equal to 2^n for n target qubits
                    raise QrackError('Invalid multiplexer instruction. Qrack only supports single qubit targets for multiplexers.')
                sim.multiplexer(operation['qubits'], len(operation['qubits']) - 1, operation['params'])
            elif name == 'reset':
                sim.reset(operation['qubits'][0])
            elif name == 'measure':
                sample_qubits.append(operation['qubits'])
                sample_clbits.append(operation['memory'])
            elif name == 'barrier':
                logger.info('Barrier gates are ignored.')
            else:
                raise QrackError('Unrecognized instruction,\'' + name + '\'')

            if len(sample_qubits) > 0:
                memory = self._add_sample_measure(sample_qubits, sample_clbits, sim, 1)
                sample_qubits = []
                sample_clbits = []

        end = time.time()

        if self._chop_threshold:
            self._statevector = sim.amplitudes().round(self._chop_threshold)
        else:
            self._statevector = sim.amplitudes()
        self._statevector = np.stack((self._statevector.real, self._statevector.imag), axis=-1)
        return {
            'name': experiment['header']['name'],
            'shots': 1,
            'data': {'statevector': self._statevector},
            'status': 'DONE',
            'success': True,
            'time_taken': (end - start),
            'header': experiment['header'],
            'metadata': 'NotImplemented'
        }

    #@profile
    def _add_sample_measure(self, sample_qubits, sample_clbits, sim, num_samples):
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

        # Get unique qubits that are actually measured
        measure_qubit = [qubit for sublist in sample_qubits for qubit in sublist]
        measure_clbit = [clbit for sublist in sample_clbits for clbit in sublist]

        # If we only want one sample, it's faster for the backend to do it,
        # without passing back the probabilities.
        if num_samples == 1:
            sample = sim.measure(measured_qubits);
            classical_state = self._classical_state
            for index in range(len(measure_qubit)):
                qubit = measure_qubit[index]
                cbit = measure_clbit[index]
                qubit_outcome = (sample & 1)
                sample = sample >> 1
                bit = 1 << cbit
                classical_state = (classical_state & (~bit)) | (qubit_outcome << cbit)
            value = bin(classical_state)[2:]
            memory.append(hex(int(value, 2)))
            return memory

        # Sample and convert to bit-strings
        measure_results = sim.measure_shots(measure_qubit, num_samples)
        classical_state = self._classical_state
        for key, value in measure_results.items():
            sample = key
            classical_state = self._classical_state
            for index in range(len(measure_qubit)):
                qubit = measure_qubit[index]
                cbit = measure_clbit[index]
                qubit_outcome = (sample & 1)
                sample = sample >> 1
                bit = 1 << cbit
                classical_state = (classical_state & (~bit)) | (qubit_outcome << cbit)
            outKey = bin(classical_state)[2:]
            memory += value * [hex(int(outKey, 2))]

        return memory

    def _validate(self, qobj, backend_options=None, noise_model=None):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.

        1. Set shots=1.
        2. Check number of qubits will fit in local memory.
        """
        name = self.name()
        if noise_model is not None:
            logger.error("{} cannot be run with a noise.".format(name))
            raise QrackError("{} does not support noise.".format(name))

        n_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if n_qubits > max_qubits:
            raise QrackError('Number of qubits ({}) '.format(n_qubits) +
                           'is greater than maximum ({}) '.format(max_qubits) +
                           'for "{}" '.format(name) +
                           'with {} GB system memory.'.format(int(local_hardware_info()['memory'])))
        if qobj.config.shots != 1:
            logger.info('"%s" only supports 1 shot. Setting shots=1.',
                        name)
            qobj.config.shots = 1
        for experiment in qobj.experiments:
            exp_name = experiment.header.name
            if getattr(experiment.config, 'shots', 1) != 1:
                logger.info('"%s" only supports 1 shot. '
                            'Setting shots=1 for circuit "%s".',
                            name, exp_name)
                experiment.config.shots = 1
