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

    * ``"normalize"`` (bool): Keep track of the total global probability
      normalization, and correct toward exactly 1. (Also turns on
      "zero_threshold". With "zero_threshold">0 "schmidt_decompose"=True,
      this can actually improve execution time, for opportune circuits.)

    * ``"zero_threshold"`` (double): Sets the threshold for truncating
      small values to zero in the simulation, gate-to-gate. (Only used
      if "normalize" is enabled. Default value: Qrack default)

    * ``"schmidt_decompose"`` (bool): If true, enable "QUnit" layer of
      Qrack, including Schmidt decomposition optimizations.

    * ``"paging"`` (bool): If true, enable "QPager" layer of Qrack.

    * ``"stabilizer"`` (bool): If true, enable Qrack "QStabilizerHybrid"
      layer of Qrack. (This can be enabled with universal gate simulations.)

    * ``"opencl"`` (bool): If true, use the OpenCL engine of Qrack
      ("QEngineOCL") as the base "Schroedinger method" simulator.
      If OpenCL is not available, simulation will fall back to CPU.

    * ``"opencl_device_id"`` (int): (If OpenCL is enabled,) choose
      the OpenCl device to simulate on, (indexed by order of device
      discovery on OpenCL load/compilation). "-1" indicates to use
      the Qrack default device, (the last discovered, which tends to
      be a non-CPU accelerator, on common personal hardware systems.)
      If "opencl-multi" is active, set the default device index.

    * ``"opencl-multi"`` (bool): (If OpenCL and Schmidt decomposition
      are enabled,) distribute Schmidt-decomposed sub-engines among
      all available OpenCL devices.
    """

    DEFAULT_CONFIGURATION = {
        'backend_name': 'statevector_simulator',
        'backend_version': __version__,
        'n_qubits': 64,
        'conditional': True,
        'url': 'https://github.com/vm6502q/qiskit-qrack-provider',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': 65536,
        'description': 'A Qrack-based, GPU-accelerated, C++ statevector simulator for qobj files',
        'coupling_map': None,
        'normalize': True,
        'zero_threshold': -999.0,
        'schmidt_decompose': True,
        'paging': True,
        'stabilizer': True,
        'opencl': True,
        'opencl_device_id': -1,
        'opencl_multi': False,
        'basis_gates': [
            'u1', 'u2', 'u3', 'u', 'r', 'p', 'cx', 'cz', 'ch', 'id', 'x', 'sx', 'y', 'z', 'h',
            'rx', 'ry', 'rz', 's', 'sdg', 't', 'tdg', 'swap', 'ccx', 'initialize', 'cu1', 'cu2',
            'cu3', 'cswap', 'mcx', 'mcy', 'mcz', 'mcu1', 'mcu2', 'mcu3', 'mcswap',
            'multiplexer', 'reset', 'measure'
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
            'name':
            'u',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional':
            True,
            'description':
            'Single-qubit gate with three rotation angles',
            'qasm_def':
            'gate u(theta,phi,lam) q { U(theta,phi,lam) q; }'
        }, {
            'name': 'p',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'Single-qubit gate [[1, 0], [0, exp(1j*lam)]]',
            'qasm_def': 'gate p(lam) q { U(0,0,lam) q; }'
        }, {
            'name': 'r',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'Single-qubit gate [[1, 0], [0, exp(1j*lam)]]',
            'qasm_def': 'gate p(lam) q { U(0,0,lam) q; }'
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
            'name': 'sx',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit square root of Pauli-X gate',
            'qasm_def': 'gate sx a { rz(-pi/2) a; h a; rz(-pi/2); }'
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
        }, {
            'name': 'reset',
            'parameters': [],
            'conditional': True,
            'description': 'Reset qubit to 0 state',
            'qasm_def': 'TODO'
        }],
        # Location where we put external libraries that will be loaded at runtime
        # by the simulator extension
        'library_dir': os.path.dirname(__file__)
    }

    # TODO: Implement these __init__ options. (We only match the signature for any compatibility at all, for now.)
    def __init__(self,
                 configuration=None,
                 provider=None,
                 seed_simulator=-1,
                 method=None,
                 max_parallel_threads=-1,
                 blocking_enable=False,
                 blocking_ignore_diagonal=False,
                 blocking_qubits=0):
        configuration = configuration or BackendConfiguration.from_dict(
            self.DEFAULT_CONFIGURATION)
        super().__init__(configuration=configuration, provider=provider)

        self._configuration = configuration
        self._number_of_qubits = None
        self._statevector = None
        self._results = {}
        self._chop_threshold = None # chop to 10^-8
        
    def options(self):
        """Return the current simulator options"""
        #return self._options
        return None

    def set_options(self, **backend_options):
        """Set the simulator options"""
        #for key, val in backend_options.items():
        #    self._set_option(key, val)

    def clear_options(self):
        """Reset the simulator options to default values."""
        #self._custom_configuration = None
        #self._custom_properties = None
        #self._custom_defaults = None
        #self._options = {}

    def run(self,
            qobj,
            backend_options=None,  # DEPRECATED
            validate=False,
            **run_options):
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
            'header': qobj.header.to_dict(),
            'metadata': {}
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
        self._statevector = 0
        self._classical_memory = 0
        self._classical_register = 0

        memory = []

        sample_qubits = []
        sample_clbits = []
        sample_cregbits = []

        start = time.time()

        try:
            sim = qrack_controller_factory()
            sim.initialize_qreg(self._configuration.opencl,
                                self._configuration.schmidt_decompose,
                                self._configuration.paging,
                                self._configuration.stabilizer,
                                self._number_of_qubits,
                                self._configuration.opencl_device_id,
                                self._configuration.opencl_multi,
                                self._configuration.normalize,
                                self._configuration.zero_threshold)
        except OverflowError:
            raise QrackError('too many qubits')

        self._measure_coalesce = True
        did_measure = False
        for operation in experiment.instructions:
            if operation.name == 'id' or operation.name == 'barrier':
                continue

            if hasattr(operation, 'conditional'):
                self._measure_coalesce = False

            if operation.name == 'measure':
                did_measure = True

        for operation in experiment.instructions:
            name = operation.name

            if name == 'id':
                logger.info('Identity gates are ignored.')
                # Skip measurement logic
                continue
            elif name == 'barrier':
                logger.info('Barrier gates are ignored.')
                # Skip measurement logic
                continue

            if (name != 'measure' or hasattr(operation, 'conditional')) and len(sample_qubits) > 0:
                if self._measure_coalesce:
                    memory = self._add_sample_measure(sample_qubits, sample_clbits, sim, shotsPerLoop)
                else:
                    self._add_qasm_measure(sample_qubits, sample_clbits, sim, sample_cregbits)
                sample_qubits = []
                sample_clbits = []
                sample_cregbits = []

            conditional = getattr(operation, 'conditional', None)
            if isinstance(conditional, int):
                conditional_bit_set = (self._classical_register >> conditional) & 1
                if not conditional_bit_set:
                    continue
            elif conditional is not None:
                mask = int(operation.conditional.mask, 16)
                if mask > 0:
                    value = self._classical_memory & mask
                    while (mask & 0x1) == 0:
                        mask >>= 1
                        value >>= 1
                    if value != int(operation.conditional.val, 16):
                        continue

            if (name == 'u1') or (name == 'p'):
                sim.u1(operation.qubits, operation.params)
            elif name == 'u2':
                sim.u2(operation.qubits, operation.params)
            elif (name == 'u3') or (name == 'u'):
                sim.u(operation.qubits, operation.params)
            elif name == 'r':
                sim.u(operation.qubits, [operation.params[0], operation.params[1] - np.pi/2, -operation.params[1] + np.pi/2])
            elif name == 'unitary':
                if (len(operation.qubits) != len(operation.params)) or (len(operation.params) != 2):
                    raise QrackError('Invalid unitary instruction. Qrack only supports single qubit targets for "unitary."')
                for qbi in range(len(operation.qubits)):
                    sim.unitary1qb([operation.qubits[qbi]], operation.params[qbi])
            elif name == 'cx':
                sim.cx(operation.qubits)
            elif name == 'cz':
                sim.cz(operation.qubits)
            elif name == 'ch':
                sim.ch(operation.qubits)
            elif name == 'x':
                sim.x(operation.qubits)
            elif name == 'sx':
                sim.sx(operation.qubits)
            elif name == 'y':
                sim.y(operation.qubits)
            elif name == 'z':
                sim.z(operation.qubits)
            elif name == 'h':
                sim.h(operation.qubits)
            elif name == 'rx':
                sim.rx(operation.qubits, operation.params)
            elif name == 'ry':
                sim.ry(operation.qubits, operation.params)
            elif name == 'rz':
                sim.rz(operation.qubits, operation.params)
            elif name == 's':
                sim.s(operation.qubits)
            elif name == 'sdg':
                sim.sdg(operation.qubits)
            elif name == 't':
                sim.t(operation.qubits)
            elif name == 'tdg':
                sim.tdg(operation.qubits)
            elif name == 'swap':
                sim.swap(operation.qubits[0], operation.qubits[1])
            elif name == 'ccx':
                sim.cx(operation.qubits)
            elif name == 'cu1':
                sim.cu1(operation.qubits, operation.params)
            elif name == 'cu2':
                sim.cu2(operation.qubits, operation.params)
            elif name == 'cu3':
                sim.cu(operation.qubits, operation.params)
            elif name == 'cswap':
                sim.cswap(operation.qubits)
            elif name == 'mcx':
                sim.cx(operation.qubits)
            elif name == 'mcy':
                sim.cy(operation.qubits)
            elif name == 'mcz':
                sim.cz(operation.qubits)
            elif name == 'initialize':
                print("initialize")
                sim.initialize(operation.qubits, operation.params)
            elif name == 'cu1':
                sim.cu1(operation.qubits, operation.params)
            elif name == 'cu2':
                sim.cu2(operation.qubits, operation.params)
            elif name == 'cu3':
                sim.cu(operation.qubits, operation.params)
            elif name == 'mcswap':
                sim.cswap(operation.qubits)
            elif name == 'multiplexer':
                if (len(operation.params) != 1 << (len(operation.qubits) - 1)) or (len(operation.params[0]) != 2):
                        raise QrackError('Invalid multiplexer instruction. Qrack only supports single qubit targets for multiplexers.')
                sim.multiplexer(operation.qubits, len(operation.qubits) - 1, operation.params)
            elif name == 'reset':
                sim.reset(operation.qubits[0])
            elif name == 'measure':
                cregbits = operation.register if hasattr(operation, 'register') else len(operation.qubits) * [-1]
                sample_qubits.append(operation.qubits)
                sample_clbits.append(operation.memory)
                sample_cregbits.append(cregbits)
            elif name == 'bfunc':
                mask = int(operation.mask, 16)
                relation = operation.relation
                val = int(operation.val, 16)

                cregbit = operation.register
                cmembit = operation.memory if hasattr(operation, 'memory') else None

                compared = (self._classical_register & mask) - val
                if relation == '==':
                    outcome = (compared == 0)
                elif relation == '!=':
                    outcome = (compared != 0)
                elif relation == '<':
                    outcome = (compared < 0)
                elif relation == '<=':
                    outcome = (compared <= 0)
                elif relation == '>':
                    outcome = (compared > 0)
                elif relation == '>=':
                    outcome = (compared >= 0)
                else:
                    raise QrackError('Invalid boolean function relation.')

                # Store outcome in register and optionally memory slot
                regbit = 1 << cregbit
                self._classical_register = \
                    (self._classical_register & (~regbit)) | (int(outcome) << cregbit)
                if cmembit is not None:
                    membit = 1 << cmembit
                    self._classical_memory = \
                        (self._classical_memory & (~membit)) | (int(outcome) << cmembit)
            else:
                backend = self.name()
                err_msg = '{0} encountered unrecognized operation "{1}"'
                raise QrackError(err_msg.format(backend, name))

        if len(sample_qubits) > 0:
            if self._measure_coalesce:
                memory = self._add_sample_measure(sample_qubits, sample_clbits, sim, 1)
            else:
                self._add_qasm_measure(sample_qubits, sample_clbits, sim, sample_cregbits)
            sample_qubits = []
            sample_clbits = []
            sample_cregbits = []

        if not self._measure_coalesce:
            # Turn classical_memory (int) into bit string and pad zero for unused cmembits
            memory.append(hex(int(bin(self._classical_memory)[2:], 2)))

        end = time.time()

        if self._chop_threshold:
            self._statevector = sim.amplitudes().round(self._chop_threshold)
        else:
            self._statevector = sim.amplitudes()
        self._statevector = np.ndarray(shape=(1<<self._number_of_qubits,), dtype='complex', buffer=np.stack((self._statevector.real, self._statevector.imag), axis=-1))
        return {
            'name': experiment.header.name,
            'shots': 1,
            'data': {'statevector': self._statevector},
            'status': 'DONE',
            'success': True,
            'time_taken': (end - start),
            'header': experiment.header.to_dict(),
            'metadata': {}
        }

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
            sample = sim.measure(measure_qubit)
            classical_state = self._classical_memory
            for index in range(len(measure_qubit)):
                qubit = measure_qubit[index]
                cbit = measure_clbit[index]
                qubit_outcome = (sample >> qubit) & 1
                bit = 1 << cbit
                classical_state = (classical_state & (~bit)) | (qubit_outcome << cbit)
            outKey = bin(classical_state)[2:]
            memory += [hex(int(outKey, 2))]
            self._classical_memory = classical_state
            return memory

        # Sample and convert to bit-strings
        measure_results = sim.measure_shots(measure_qubit, num_samples)
        classical_state = self._classical_memory
        for key, value in measure_results.items():
            sample = key
            classical_state = self._classical_memory
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

    def _add_qasm_measure(self, sample_qubits, sample_clbits, sim, sample_cregbits=None):
        """Apply a measure instruction to a qubit.
        Args:
            qubit (int): qubit is the qubit measured.
            cmembit (int): is the classical memory bit to store outcome in.
            cregbit (int, optional): is the classical register bit to store outcome in.
        """

        measure_qubit = [qubit for sublist in sample_qubits for qubit in sublist]
        measure_clbit = [clbit for sublist in sample_clbits for clbit in sublist]
        measure_cregbit = [clbit for sublist in sample_cregbits for clbit in sublist]

        sample = sim.measure(measure_qubit)
        classical_state = self._classical_memory
        classical_register = self._classical_register
        for index in range(len(measure_qubit)):
            qubit = measure_qubit[index]
            cbit = measure_clbit[index]
            cregbit = measure_cregbit[index]
            qubit_outcome = (sample >> qubit) & 1
            bit = 1 << cbit
            classical_state = (classical_state & (~bit)) | (qubit_outcome << cbit)
            if cregbit >= 0:
                regbit = 1 << cregbit
                classical_register = \
                    (classical_register & (~regbit)) | (int(qubit_outcome) << cregbit)
        self._classical_memory = classical_state
        self._classical_register = classical_register

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
