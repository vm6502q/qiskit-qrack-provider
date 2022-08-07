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


import os
import math
import uuid
import time
import random
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

class NoisyCliffordTSimulator(BackendV1):
    """
    A very efficient Clifford+T simulator with an intrinsically noisy T gate (and inverse).
    """

    DEFAULT_OPTIONS = {
        'method': 'automatic',
        'shots': 1024,
        'noise': 0.0004,
        'is_schmidt_decompose_multi': True,
        'is_schmidt_decompose': True,
        'is_stabilizer_hybrid': True,
        'is_binary_decision_tree': False,
        'is_paged': True,
        'is_1qb_fusion': False,
        'is_cpu_gpu_hybrid': True,
        'is_host_pointer': False,
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
            'id', 'h', 'x', 'y', 'z', 's', 'sdg', 'sx', 'sxdg', 't', 'tdg', 'rz', 'u1', 'p'
            'cx', 'cy', 'cz', 'swap', 'iswap', 'reset', 'measure', 'barrier'
        ],
        'gates': [{
            'name': 'id',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit identity gate',
            'qasm_def': 'gate id a { U(0,0,0) a; }'
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
            'name': 'rz',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-Z axis rotation gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'u1',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'Single-qubit gate [[1, 0], [0, exp(1j*lam)]]',
            'qasm_def': 'gate u1(lam) q { U(0,0,lam) q; }'
        }, {
            'name': 'p',
            'parameters': ['theta', 'phi'],
            'conditional': True,
            'description': 'Single-qubit gate [[cos(theta), -1j*exp(-1j*phi)], [sin(theta), -1j*exp(1j *phi)*sin(theta), cos(theta)]]',
            'qasm_def': 'gate r(theta, phi) q { U(theta, phi - pi/2, -phi + pi/2) q;}'
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
            'isSchmidtDecomposeMulti': options.is_schmidt_decompose_multi if hasattr(options, 'is_schmidt_decompose_multi') else self._options.get('is_schmidt_decompose_multi'),
            'isSchmidtDecompose': options.is_schmidt_decompose if hasattr(options, 'is_schmidt_decompose') else self._options.get('is_schmidt_decompose'),
            'isStabilizerHybrid': options.is_stabilizer_hybrid if hasattr(options, 'is_stabilizer_hybrid') else self._options.get('is_stabilizer_hybrid'),
            'isBinaryDecisionTree': options.is_binary_decision_tree if hasattr(options, 'is_binary_decision_tree') else self._options.get('is_binary_decision_tree'),
            'isPaged': options.is_paged if hasattr(options, 'is_paged') else self._options.get('is_paged'),
            'is1QbFusion': options.is_1qb_fusion if hasattr(options, 'is_1qb_fusion') else self._options.get('is_1qb_fusion'),
            'isCpuGpuHybrid': options.is_cpu_gpu_hybrid if hasattr(options, 'is_cpu_gpu_hybrid') else self._options.get('is_cpu_gpu_hybrid'),
            'isHostPointer': options.is_host_pointer if hasattr(options, 'is_host_pointer') else self._options.get('is_host_pointer'),
        }

        # In the ideal, if the user isn't demanding a specific separability threshold, this is a well-motivated choice.
        noise = options.noise if hasattr(options, 'noise') else self._options.get('noise')
        self._reset_separability_threshold = False
        if "QRACK_QUNIT_SEPARABILITY_THRESHOLD" not in os.environ:
            self._reset_separability_threshold = True
            os.environ["QRACK_QUNIT_SEPARABILITY_THRESHOLD"] = str(noise / 2)

        data = run_input.config.memory if hasattr(run_input, 'config') else []
        self._shots = options['shots'] if 'shots' in options else (run_input.config.shots if hasattr(run_input, 'config') else self._options.get('shots'))
        qobj_id = options['qobj_id'] if 'qobj_id' in options else (run_input.qobj_id if hasattr(run_input, 'config') else '')
        qobj_header = options['qobj_header'] if 'qobj_header' in options else (run_input.header if hasattr(run_input, 'config') else {})
        job_id = str(uuid.uuid4())

        job = QrackJob(self, job_id, self._run_job(job_id, run_input, data, qobj_id, qobj_header, noise, **qrack_options), run_input)
        return job

    def _run_job(self, job_id, run_input, data, qobj_id, qobj_header, noise, **options):
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
            results.append(self.run_experiment(experiment, noise, **options))

        if self._reset_separability_threshold:
            # If the separability threshold is auto-default, clear its automatically assigned environment variable.
            del os.environ["QRACK_QUNIT_SEPARABILITY_THRESHOLD"]

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

    def run_experiment(self, experiment, noise, **options):
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

        shotsPerLoop = self._shots
        shotLoopMax = 1

        is_initializing = True

        for shot in range(shotLoopMax):
            self._sim = QrackSimulator(qubitCount = self._number_of_qubits, **options)
            self._classical_memory = 0
            self._classical_register = 0

            for operation in instructions:
                self._apply_op(operation, noise)

            if len(self._sample_qubits) > 0:
                self._data += [hex(int(bin(self._classical_memory)[2:], 2))]
                self._sample_qubits = []
                self._sample_clbits = []
                self._sample_cregbits = []

        data = { 'counts': dict(Counter(self._data)) }
        if isinstance(experiment, QasmQobjExperiment):
            data['memory'] = self._data
            data = QrackExperimentResultData(**data)
        else:
            data = pd.DataFrame(data=data)

        metadata = { 'measure_sampling': False }
        if isinstance(experiment, QuantumCircuit) and hasattr(experiment, 'metadata'):
            metadata = experiment.metadata

        return QrackExperimentResult(
            shots = self._shots,
            data = data,
            status = 'DONE',
            success = True,
            header = experiment.header if isinstance(experiment, QasmQobjExperiment) else QrackExperimentResultHeader(name = experiment.name),
            meta_data = metadata,
            time_taken = (time.time() - start)
        )

    def inject_depolarizing_1qb_noise(self, qubit, lam):
        if (lam <= 0.):
            return

        # Random azimuth around Z axis of measurement,
        # which effectively randomizes depolarization orientation in X-Y plane
        angleZ = random.uniform(0., 4. * math.pi)
        # Total depolarization magnitude angle,
        # where use of hyperbolic trig. is based on empirical exploratory work
        lamAngle = math.asinh(2 * lam)

        # Azimuth
        self._sim.r(Pauli.PauliZ, angleZ, qubit)
        # Depolarization magnitude
        self._sim.r(Pauli.PauliY, lamAngle, qubit)

        # Allocate an ancilla
        ancilla = self._sim.num_qubits()
        self._sim.allocate_qubit(ancilla)
        # Partially entangle with the ancilla
        self._sim.mcx([qubit], ancilla)
        # Partially collapse the original state
        self._sim.m(ancilla)
        # The ancilla is fully separable, after measurement.
        self._sim.release(ancilla)

        # Uncompute
        self._sim.r(Pauli.PauliY, -lamAngle, qubit)
        self._sim.r(Pauli.PauliZ, -angleZ, qubit)

    def _apply_op(self, operation, noise):
        name = operation.name

        # Divert variational phase gates to Clifford, when possible.
        if name == 'rz' or name == 'u1' or name == 'p':
            angle = operation.params[0]
            while angle < 0.:
                angle = angle + 2. * math.pi
            while angle >= 2. * math.pi:
                angle = angle - 2. * math.pi
            if math.isclose(angle, 0.):
                return
            if math.isclose(angle, math.pi):
                name = 'z'
            elif math.isclose(angle, math.pi / 2.):
                name = 's'
            elif math.isclose(angle, -math.pi / 2.):
                name = 'sdg'

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

        if name == 'h':
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
            self.inject_depolarizing_1qb_noise(operation.qubits[0], noise)
        elif name == 'tdg':
            self._sim.adjt(operation.qubits[0])
            self.inject_depolarizing_1qb_noise(operation.qubits[0], noise)
        elif name == 'rz':
            self._sim.r(Pauli.PauliZ, operation.params[0], operation.qubits[0])
            self.inject_depolarizing_1qb_noise(operation.qubits[0], noise)
        elif (name == 'u1') or (name == 'p'):
            self._sim.u(operation.qubits[0], 0, 0, operation.params[0])
            self.inject_depolarizing_1qb_noise(operation.qubits[0], noise)
        elif name == 'cx':
            self._sim.mcx(operation.qubits[0:1], operation.qubits[1])
        elif name == 'cy':
            self._sim.mcy(operation.qubits[0:1], operation.qubits[1])
        elif name == 'cz':
            self._sim.mcz(operation.qubits[0:1], operation.qubits[1])
        elif name == 'swap':
            self._sim.swap(operation.qubits[0], operation.qubits[1])
        elif name == 'iswap':
            self._sim.iswap(operation.qubits[0], operation.qubits[1])
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

    @staticmethod
    def name():
        return 'noisy_clifford_t_simulator'
