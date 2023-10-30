# This code is based on and adapted from https://github.com/Qiskit/qiskit-qcgpu-provider/blob/master/qiskit_qcgpu_provider/qasm_simulator.py
#
# Adapted by Daniel Strano. Many thanks to Adam Kelly for pioneering a third-party Qiskit provider.
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
import pandas as pd
from datetime import datetime
from collections import Counter
from qiskit.providers.models import BackendConfiguration

from ..version import __version__
from ..qrackjob import QrackJob
from ..qrackerror import QrackError
from pyqrack import QrackSimulator, Pauli

from qiskit.providers.backend import BackendV1
from qiskit.result import Result
from qiskit.providers.options import Options

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.qobj.qasm_qobj import QasmQobjExperiment, QasmQobjInstruction
from qiskit.circuit.classicalregister import Clbit


class QrackQasmQobjInstructionConditional:
    def __init__(self, mask, val):
        self.mask = mask
        self.val = val

    def to_dict(self):
        return vars(self)


class QrackExperimentHeader(dict):
    def __init__(self, a_dict=None):
        dict.__init__(self)
        for key, value in a_dict.items():
            self[key] = value

    def to_dict(self):
        return self

class QrackExperimentResultHeader:
    def __init__(self, name):
        self.name = name

    def to_dict(self):
        return vars(self)


class QrackExperimentResultData:
    def __init__(self, counts, memory):
        self.counts = counts
        self.memory = memory

    def to_dict(self):
        return vars(self)


class QrackExperimentResult:
    def __init__(self, shots, data, status, success, header, meta_data = None, time_taken = None):
        self.shots = shots
        self.data = data
        self.status = status
        self.success = success
        self.header = header
        self.meta_data = meta_data,
        self.time_taken = time_taken

    def to_dict(self):
        return vars(self)


class QasmSimulator(BackendV1):
    """
    Contains an OpenCL based backend
    """

    DEFAULT_OPTIONS = {
        'method': 'automatic',
        'shots': 1024,
        'is_tensor_network': True,
        'is_schmidt_decompose_multi': True,
        'is_schmidt_decompose': True,
        'is_stabilizer_hybrid': True,
        'is_binary_decision_tree': False,
        'is_paged': True,
        'is_cpu_gpu_hybrid': True,
        'is_host_pointer': False,
        'is_t_injected': True,
        'is_reactively_separated': True
    }

    DEFAULT_CONFIGURATION = {
        'backend_name': 'statevector_simulator',
        'backend_version': __version__,
        'n_qubits': 64,
        'conditional': True,
        'url': 'https://github.com/vm6502q/qiskit-qrack-provider',
        'simulator': True,
        'local': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': 65536,
        'description': 'An OpenCL based qasm simulator',
        'coupling_map': None,
        'basis_gates': [
            'id', 'u', 'u1', 'u2', 'u3', 'r', 'rx', 'ry', 'rz',
            'h', 'x', 'y', 'z', 's', 'sdg', 'sx', 'sxdg', 'p', 't', 'tdg',
            'cu', 'cu1', 'cu2', 'cu3', 'cx', 'cy', 'cz', 'ch', 'cp', 'csx', 'csxdg', 'dcx',
            'ccx', 'ccy', 'ccz', 'mcx', 'mcy', 'mcz', 'mcu', 'mcu1', 'mcu2', 'mcu3',
            'swap', 'iswap', 'cswap', 'mcswap', 'reset', 'measure', 'barrier'
        ],
        'gates': [{
            'name': 'id',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit identity gate',
            'qasm_def': 'gate id a { U(0,0,0) a; }'
        }, {
            'name': 'u',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional': True,
            'description': 'Single-qubit gate with three rotation angles',
            'qasm_def': 'gate u(theta,phi,lam) q { U(theta,phi,lam) q; }'
        }, {
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
            'name': 'u3',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional': True,
            'description': 'Single-qubit gate with three rotation angles',
            'qasm_def': 'gate u3(theta,phi,lam) q { U(theta,phi,lam) q; }'
        }, {
            'name': 'r',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'Single-qubit gate [[1, 0], [0, exp(1j*lam)]]',
            'qasm_def': 'gate p(lam) q { U(0,0,lam) q; }'
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
            'name': 'h',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Hadamard gate',
            'qasm_def': 'TODO'
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
            'name': 'sx',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit square root of Pauli-X gate',
            'qasm_def': 'gate sx a { rz(-pi/2) a; h a; rz(-pi/2); }'
        }, {
            'name': 'sxdg',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit inverse square root of Pauli-X gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'p',
            'parameters': ['theta', 'phi'],
            'conditional': True,
            'description': 'Single-qubit gate [[cos(theta), -1j*exp(-1j*phi)], [sin(theta), -1j*exp(1j *phi)*sin(theta), cos(theta)]]',
            'qasm_def': 'gate r(theta, phi) q { U(theta, phi - pi/2, -phi + pi/2) q;}'
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
            'name': 'cu',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional': True,
            'description': 'Two-qubit Controlled-u gate',
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
            'name': 'cx',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit Controlled-NOT gate',
            'qasm_def': 'gate cx c,t { CX c,t; }'
        }, {
            'name': 'cy',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit Controlled-Y gate',
            'qasm_def': 'TODO'
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
            'name': 'cp',
            'parameters': [],
            'conditional': True,
            'description': 'Controlled-Phase gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'csx',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit Controlled square root of Pauli-X gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'csxdg',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit controlled inverse square root of Pauli-X gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'dcx',
            'parameters': [],
            'conditional': True,
            'description': 'Double-CNOT gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'ccx',
            'parameters': [],
            'conditional': True,
            'description': 'Three-qubit Toffoli gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'ccy',
            'parameters': [],
            'conditional': True,
            'description': 'Three-qubit controlled-Y gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'ccz',
            'parameters': [],
            'conditional': True,
            'description': 'Three-qubit controlled-Z gate',
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
            'name': 'mcu',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional': True,
            'description': 'N-qubit multi-controlled-u3 gate',
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
            'name': 'swap',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit SWAP gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'iswap',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit ISWAP gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'cswap',
            'parameters': [],
            'conditional': True,
            'description': 'Three-qubit Fredkin (controlled-SWAP) gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcswap',
            'parameters': [],
            'conditional': True,
            'description': 'N-qubit multi-controlled-SWAP gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'measure',
            'parameters': [],
            'conditional': True,
            'description': 'Measure qubit',
            'qasm_def': 'TODO'
        }, {
            'name': 'reset',
            'parameters': [],
            'conditional': True,
            'description': 'Reset qubit to 0 state',
            'qasm_def': 'TODO'
        }, {
            'name': 'barrier',
            'parameters': [],
            'conditional': True,
            'description': 'Barrier primitive for quantum circuit',
            'qasm_def': 'TODO'
        }]
    }

    def __init__(self, configuration=None, provider=None, **fields):
        """Initialize a backend class

        Args:
            configuration (BackendConfiguration): A backend configuration
                object for the backend object.
            provider (qiskit.providers.Provider): Optionally, the provider
                object that this Backend comes from.
            fields: kwargs for the values to use to override the default
                options.
        Raises:
            AttributeError: if input field not a valid options

        ..
            This next bit is necessary just because autosummary generally won't summarise private
            methods; changing that behaviour would have annoying knock-on effects through all the
            rest of the documentation, so instead we just hard-code the automethod directive.

        In addition to the public abstract methods, subclasses should also implement the following
        private methods:

        .. automethod:: _default_options
        """

        configuration = configuration or BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION)

        self._number_of_qubits = 0
        self._number_of_clbits = 0
        self._shots = 1

        self._configuration = configuration
        self._options = self._default_options()
        self._provider = provider
        if fields:
            for field in fields:
                if  field not in self.DEFAULT_OPTIONS:
                    raise AttributeError("Options field %s is not valid for this backend" % field)
            self._options.update_options(**fields)

    @classmethod
    def _default_options(cls):
        """Return the default options

        This method will return a :class:`qiskit.providers.Options`
        subclass object that will be used for the default options. These
        should be the default parameters to use for the options of the
        backend.

        Returns:
            qiskit.providers.Options: A options object with
                default values set
        """
        # WARNING: The above prototype for return type doesn't work in BackEndV1 in Qiskit v0.30.0.
        # We're resorting to duck typing.
        _def_opts = Options()
        _def_opts.update_options(**cls.DEFAULT_OPTIONS)
        return _def_opts

    def run(self, run_input, **options):
        """Run on the backend.

        This method that will return a :class:`~qiskit.providers.Job` object
        that run circuits. Depending on the backend this may be either an async
        or sync call. It is the discretion of the provider to decide whether
        running should  block until the execution is finished or not. The Job
        class can handle either situation.

        Args:
            run_input (QuantumCircuit or Schedule or list): An individual or a
                list of :class:`~qiskit.circuits.QuantumCircuit` or
                :class:`~qiskit.pulse.Schedule` objects to run on the backend.
                For legacy providers migrating to the new versioned providers,
                provider interface a :class:`~qiskit.qobj.QasmQobj` or
                :class:`~qiskit.qobj.PulseQobj` objects should probably be
                supported too (but deprecated) for backwards compatibility. Be
                sure to update the docstrings of subclasses implementing this
                method to document that. New provider implementations should not
                do this though as :mod:`qiskit.qobj` will be deprecated and
                removed along with the legacy providers interface.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object then the expectation is that the value
                specified will be used instead of what's set in the options
                object.
        Returns:
            Job: The job object for the run
        """

        qrack_options = {
            'isTensorNetwork': options.is_tensor_network if hasattr(options, 'is_tensor_network') else self._options.get('is_tensor_network'),
            'isSchmidtDecomposeMulti': options.is_schmidt_decompose_multi if hasattr(options, 'is_schmidt_decompose_multi') else self._options.get('is_schmidt_decompose_multi'),
            'isSchmidtDecompose': options.is_schmidt_decompose if hasattr(options, 'is_schmidt_decompose') else self._options.get('is_schmidt_decompose'),
            'isStabilizerHybrid': options.is_stabilizer_hybrid if hasattr(options, 'is_stabilizer_hybrid') else self._options.get('is_stabilizer_hybrid'),
            'isBinaryDecisionTree': options.is_binary_decision_tree if hasattr(options, 'is_binary_decision_tree') else self._options.get('is_binary_decision_tree'),
            'isPaged': options.is_paged if hasattr(options, 'is_paged') else self._options.get('is_paged'),
            'isCpuGpuHybrid': options.is_cpu_gpu_hybrid if hasattr(options, 'is_cpu_gpu_hybrid') else self._options.get('is_cpu_gpu_hybrid'),
            'isHostPointer': options.is_host_pointer if hasattr(options, 'is_host_pointer') else self._options.get('is_host_pointer'),
        }

        data = run_input.config.memory if hasattr(run_input, 'config') else []
        self._shots = options['shots'] if 'shots' in options else (run_input.config.shots if hasattr(run_input, 'config') else self._options.get('shots'))
        self.is_t = options.is_t_injected if hasattr(options, 'is_t_injected') else self._options.get('is_t_injected')
        self.is_reactive = options.is_reactively_separated if hasattr(options, 'is_reactively_separated') else self._options.get('is_reactively_separated')
        qobj_id = options['qobj_id'] if 'qobj_id' in options else (run_input.qobj_id if hasattr(run_input, 'config') else '')
        qobj_header = options['qobj_header'] if 'qobj_header' in options else (run_input.header if hasattr(run_input, 'config') else {})
        job_id = str(uuid.uuid4())

        job = QrackJob(self, job_id, self._run_job(job_id, run_input, data, qobj_id, qobj_header, **qrack_options), run_input)
        return job

    def _run_job(self, job_id, run_input, data, qobj_id, qobj_header, **options):
        """Run experiments in run_input
        Args:
            job_id (str): unique id for the job.
            run_input (QuantumCircuit or Schedule or list): job description
        Returns:
            Result: Result object
        """
        start = time.time()

        self._data = data

        experiments = run_input.experiments if hasattr(run_input, 'config') else run_input
        if isinstance(experiments, QuantumCircuit):
            experiments = [experiments]
        results = []

        for experiment in experiments:
            results.append(self.run_experiment(experiment, **options))

        return Result(
            backend_name = self.name(),
            backend_version = self._configuration.backend_version,
            qobj_id = qobj_id,
            job_id = job_id,
            success = True,
            results = results,
            date = datetime.now(),
            status = 'COMPLETED',
            header = QrackExperimentHeader(qobj_header) if type(qobj_header) is dict else qobj_header,
            time_taken = (time.time() - start)
        )

    def run_experiment(self, experiment, **options):
        """Run an experiment (circuit) and return a single experiment result.
        Args:
            experiment (QobjExperiment): experiment from qobj experiments list
        Returns:
            dict: A dictionary of results.
            dict: A result dictionary
        Raises:
            QrackError: If the number of qubits is too large, or another
                error occurs during execution.
        """
        start = time.time()

        instructions = []
        if isinstance(experiment, QasmQobjExperiment):
            self._number_of_qubits = experiment.header.n_qubits
            self._number_of_clbits = experiment.header.memory_slots
            instructions = experiment.instructions
        elif isinstance(experiment, QuantumCircuit):
            self._number_of_qubits = len(experiment.qubits)
            self._number_of_clbits = len(experiment.clbits)
            for datum in experiment._data:
                qubits = []
                for qubit in datum[1]:
                    qubits.append(experiment.qubits.index(qubit))

                clbits = []
                for clbit in datum[2]:
                    clbits.append(experiment.clbits.index(clbit))

                conditional = None
                condition = datum[0].condition
                if condition is not None:
                    if isinstance(condition[0], Clbit):
                        conditional = experiment.clbits.index(condition[0])
                    else:
                        creg_index = experiment.cregs.index(condition[0])
                        size = experiment.cregs[creg_index].size
                        offset = 0
                        for i in range(creg_index):
                            offset += len(experiment.cregs[i])
                        mask = ((1 << offset) - 1) ^ ((1 << (offset + size)) - 1)
                        val = condition[1]
                        conditional = offset if (size == 1) else QrackQasmQobjInstructionConditional(mask, val)

                instructions.append(QasmQobjInstruction(
                    datum[0].name,
                    qubits = qubits,
                    memory = clbits,
                    condition=condition,
                    conditional=conditional,
                    params = datum[0].params
                ))
        else:
            raise QrackError('Unrecognized "run_input" argument specified for run().')

        self._sample_qubits = []
        self._sample_clbits = []
        self._sample_cregbits = []
        self._data = []

        self._sample_measure = True
        shotsPerLoop = self._shots
        shotLoopMax = 1

        is_initializing = True
        boundary_start = -1

        for opcount in range(len(instructions)):
            operation = instructions[opcount]

            if operation.name == 'id' or operation.name == 'barrier':
                continue

            if is_initializing and ((operation.name == 'measure') or (operation.name == 'reset')):
                continue

            is_initializing = False

            if (operation.name == 'measure') or (operation.name == 'reset'):
                if boundary_start == -1:
                    boundary_start = opcount

            if (boundary_start != -1) and (operation.name != 'measure'):
                shotsPerLoop = 1
                shotLoopMax = self._shots
                self._sample_measure = False
                break

        preamble_memory = 0
        preamble_register = 0
        preamble_sim = None

        if self._sample_measure or boundary_start <= 0:
            boundary_start = 0
            self._sample_measure = True
            shotsPerLoop = self._shots
            shotLoopMax = 1
        else:
            boundary_start -= 1
            if boundary_start > 0:
                self._sim = QrackSimulator(qubitCount = self._number_of_qubits, **options)
                self._sim.set_t_injection(self.is_t)
                self._sim.set_reactive_separate(self.is_reactive)
                self._classical_memory = 0
                self._classical_register = 0

                for operation in instructions[:boundary_start]:
                    self._apply_op(operation)

                preamble_memory = self._classical_memory
                preamble_register = self._classical_register
                preamble_sim = self._sim

        for shot in range(shotLoopMax):
            if preamble_sim is None:
                self._sim = QrackSimulator(qubitCount = self._number_of_qubits, **options)
                self._sim.set_t_injection(self.is_t)
                self._sim.set_reactive_separate(self.is_reactive)
                self._classical_memory = 0
                self._classical_register = 0
            else:
                self._sim = QrackSimulator(cloneSid = preamble_sim.sid)
                self._sim.set_t_injection(self.is_t)
                self._sim.set_reactive_separate(self.is_reactive)
                self._classical_memory = preamble_memory
                self._classical_register = preamble_register

            for operation in instructions[boundary_start:]:
                self._apply_op(operation)

            if not self._sample_measure and (len(self._sample_qubits) > 0):
                self._data += [hex(int(bin(self._classical_memory)[2:], 2))]
                self._sample_qubits = []
                self._sample_clbits = []
                self._sample_cregbits = []

        if self._sample_measure and (len(self._sample_qubits) > 0):
            self._data = self._add_sample_measure(self._sample_qubits, self._sample_clbits, self._shots)

        data = { 'counts': dict(Counter(self._data)) }
        if isinstance(experiment, QasmQobjExperiment):
            data['memory'] = self._data
            data = QrackExperimentResultData(**data)
        else:
            data = pd.DataFrame(data=data)

        metadata = { 'measure_sampling': self._sample_measure }
        if isinstance(experiment, QuantumCircuit) and hasattr(experiment, 'metadata') and experiment.metadata:
            metadata = experiment.metadata
            metadata['measure_sampling'] = self._sample_measure

        return QrackExperimentResult(
            shots = self._shots,
            data = data,
            status = 'DONE',
            success = True,
            header = experiment.header if isinstance(experiment, QasmQobjExperiment) else QrackExperimentResultHeader(name = experiment.name),
            meta_data = metadata,
            time_taken = (time.time() - start)
        )

    def _apply_op(self, operation):
        name = operation.name

        if (name == 'id') or (name == 'barrier'):
            # Skip measurement logic
            return

        conditional = getattr(operation, 'conditional', None)
        if isinstance(conditional, int):
            conditional_bit_set = (self._classical_register >> conditional) & 1
            if not conditional_bit_set:
                return
        elif conditional is not None:
            mask = int(conditional.mask, 16)
            if mask > 0:
                value = self._classical_memory & mask
                while (mask & 0x1) == 0:
                    mask >>= 1
                    value >>= 1
                if value != int(conditional.val, 16):
                    return

        if (name == 'u1') or (name == 'p'):
            self._sim.u(operation.qubits[0], 0, 0, float(operation.params[0]))
        elif name == 'u2':
            self._sim.u(operation.qubits[0], np.pi / 2, float(operation.params[0]), float(operation.params[1]))
        elif (name == 'u3') or (name == 'u'):
            self._sim.u(operation.qubits[0], float(operation.params[0]), float(operation.params[1]), float(operation.params[2]))
        elif name == 'r':
            self._sim.u(operation.qubits[0], float(operation.params[0]), float(operation.params[1]) - np.pi/2, (-1 * float(operation.params[1])) + np.pi/2)
        elif (name == 'unitary') and (len(operation.qubits) == 1):
            self._sim.mtrx(operation.params[0].flatten(), operation.qubits[0])
        elif name == 'rx':
            self._sim.r(Pauli.PauliX, float(operation.params[0]), operation.qubits[0])
        elif name == 'ry':
            self._sim.r(Pauli.PauliY, float(operation.params[0]), operation.qubits[0])
        elif name == 'rz':
            self._sim.r(Pauli.PauliZ, float(operation.params[0]), operation.qubits[0])
        elif name == 'h':
            self._sim.h(operation.qubits[0])
        elif name == 'x':
            self._sim.x(operation.qubits[0])
        elif name == 'y':
            self._sim.y(operation.qubits[0])
        elif name == 'z':
            self._sim.z(operation.qubits[0])
        elif name == 's':
            self._sim.s(operation.qubits[0])
        elif name == 'sdg':
            self._sim.adjs(operation.qubits[0])
        elif name == 'sx':
            self._sim.mtrx([(1+1j)/2, (1-1j)/2, (1-1j)/2, (1+1j)/2], operation.qubits[0])
        elif name == 'sxdg':
            self._sim.mtrx([(1-1j)/2, (1+1j)/2, (1+1j)/2, (1-1j)/2], operation.qubits[0])
        elif name == 't':
            self._sim.t(operation.qubits[0])
        elif name == 'tdg':
            self._sim.adjt(operation.qubits[0])
        elif name == 'cu1':
            self._sim.mcu(operation.qubits[0:1], operation.qubits[1], 0, 0, float(operation.params[0]))
        elif name == 'cu2':
            self._sim.mcu(operation.qubits[0:1], operation.qubits[1], np.pi / 2, float(operation.params[0]), float(operation.params[1]))
        elif (name == 'cu3') or (name == 'cu'):
            self._sim.mcu(operation.qubits[0:1], operation.qubits[1], float(operation.params[0]), float(operation.params[1]), float(operation.params[2]))
        elif name == 'mcu1':
            self._sim.mcu(operation.qubits[0:-1], operation.qubits[-1], 0, 0, float(operation.params[0]))
        elif name == 'mcu2':
            self._sim.mcu(operation.qubits[0:-1], operation.qubits[-1], np.pi / 2, float(operation.params[0]), float(operation.params[1]))
        elif (name == 'mcu3') or (name == 'mcu'):
            self._sim.mcu(operation.qubits[0:-1], operation.qubits[-1], float(operation.params[0]), float(operation.params[1]), float(operation.params[2]))
        elif name == 'cx':
            self._sim.mcx(operation.qubits[0:1], operation.qubits[1])
        elif name == 'cy':
            self._sim.mcy(operation.qubits[0:1], operation.qubits[1])
        elif name == 'cz':
            self._sim.mcz(operation.qubits[0:1], operation.qubits[1])
        elif name == 'ch':
            self._sim.mch(operation.qubits[0:1], operation.qubits[1])
        elif name == 'cp':
            self._sim.mcmtrx(operation.qubits[0:1], [1, 0, 0, np.exp(1j * float(operation.params[0]))], operation.qubits[1])
        elif name == 'csx':
            self._sim.mcmtrx(operation.qubits[0:1], [(1+1j)/2, (1-1j)/2, (1-1j)/2, (1+1j)/2], operation.qubits[1])
        elif name == 'csxdg':
            self._sim.mcmtrx(operation.qubits[0:1], [(1-1j)/2, (1+1j)/2, (1+1j)/2, (1-1j)/2], operation.qubits[1])
        elif name == 'dcx':
            self._sim.mcx(operation.qubits[0:1], operation.qubits[1])
            self._sim.mcx(operation.qubits[1:2], operation.qubits[0])
        elif name == 'ccx':
            self._sim.mcx(operation.qubits[0:2], operation.qubits[2])
        elif name == 'ccy':
            self._sim.mcy(operation.qubits[0:2], operation.qubits[2])
        elif name == 'ccz':
            self._sim.mcz(operation.qubits[0:2], operation.qubits[2])
        elif name == 'mcx':
            self._sim.mcx(operation.qubits[0:-1], operation.qubits[-1])
        elif name == 'mcy':
            self._sim.mcy(operation.qubits[0:-1], operation.qubits[-1])
        elif name == 'mcz':
            self._sim.mcz(operation.qubits[0:-1], operation.qubits[-1])
        elif name == 'swap':
            self._sim.swap(operation.qubits[0], operation.qubits[1])
        elif name == 'iswap':
            self._sim.iswap(operation.qubits[0], operation.qubits[1])
        elif name == 'cswap':
            self._sim.cswap(operation.qubits[0:1], operation.qubits[1], operation.qubits[2])
        elif name == 'mcswap':
            self._sim.cswap(operation.qubits[:-2], operation.qubits[-2], operation.qubits[-1])
        elif name == 'reset':
            qubits = operation.qubits
            for qubit in qubits:
                if self._sim.m(qubit):
                    self._sim.x(qubit)
        elif name == 'measure':
            qubits = operation.qubits
            clbits = operation.memory
            cregbits = operation.register if hasattr(operation, 'register') else len(operation.qubits) * [-1]

            self._sample_qubits += qubits
            self._sample_clbits += clbits
            self._sample_cregbits += cregbits

            if not self._sample_measure:
                for index in range(len(qubits)):
                    qubit_outcome = self._sim.m(qubits[index])

                    clbit = clbits[index]
                    clmask = 1 << clbit
                    self._classical_memory = (self._classical_memory & (~clmask)) | (qubit_outcome << clbit)

                    cregbit = cregbits[index]
                    if cregbit < 0:
                        cregbit = clbit

                    regbit = 1 << cregbit
                    self._classical_register = (self._classical_register & (~regbit)) | (qubit_outcome << cregbit)

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
            raise QrackError(err_msg.format(backend, operation))

    def _add_sample_measure(self, sample_qubits, sample_clbits, num_samples):
        """Generate data samples from current statevector.
        Taken almost straight from the terra source code.
        Args:
            measure_params (list): List of (qubit, clbit) values for
                                   measure instructions to sample.
            num_samples (int): The number of data samples to generate.
        Returns:
            list: A list of data values in hex format.
        """
        # Get unique qubits that are actually measured
        measure_qubit = [qubit for qubit in sample_qubits]
        measure_clbit = [clbit for clbit in sample_clbits]

        # Sample and convert to bit-strings
        data = []
        if num_samples == 1:
            sample = self._sim.m_all()
            result = 0
            for index in range(len(measure_qubit)):
                qubit = measure_qubit[index]
                qubit_outcome = ((sample >> qubit) & 1)
                result |= qubit_outcome << index
            measure_results = [result]
        else:
            measure_results = self._sim.measure_shots(measure_qubit, num_samples)

        for sample in measure_results:
            for index in range(len(measure_qubit)):
                qubit_outcome = ((sample >> index) & 1)
                clbit = measure_clbit[index]
                clmask = 1 << clbit
                self._classical_memory = (self._classical_memory & (~clmask)) | (qubit_outcome << clbit)

            data.append(hex(int(bin(self._classical_memory)[2:], 2)))

        return data

    @staticmethod
    def name():
        return 'qasm_simulator'
