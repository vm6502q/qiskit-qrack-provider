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
import math
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

from ..version import __version__
from ..qrackjob import QrackJob
from ..qrackerror import QrackError
from pyqrack import QrackAceBackend, Pauli

from qiskit.providers.backend import BackendV2
from qiskit.result import Result
from qiskit.providers.options import Options
from qiskit.transpiler import Target, CouplingMap
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import Clbit
from qiskit.circuit import Parameter
from qiskit.transpiler import CouplingMap

from qiskit.circuit.library import IGate, U3Gate, U2Gate, U1Gate, XGate, YGate, ZGate, HGate, RXGate, RYGate, RZGate, SGate, SdgGate, TGate, TdgGate, CXGate, CYGate, CZGate, SwapGate, iSwapGate

from qiskit_aer.noise import NoiseModel, depolarizing_error


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

    def get(self, param, val):
        return self.name if param == 'name' else val

    def items(self):
        return { 'name': self.name }.items()

class QrackExperimentResultData:
    def __init__(self, counts, memory):
        self.counts = counts
        self.memory = memory

    def to_dict(self):
        return vars(self)


class QrackExperimentResult:
    def __init__(self, shots, data, status, success, header, metadata = None, time_taken = None):
        self.shots = shots
        self.data = data
        self.status = status
        self.success = success
        self.header = header
        self.metadata = metadata,
        self.time_taken = time_taken

    def to_dict(self):
        return vars(self)


class AceQasmSimulator(BackendV2):
    """
    Contains a fundamentally-optimized, approximate, nearest-neighbor backend
    """

    DEFAULT_OPTIONS = {
        'method': 'matrix_product_state',
        'n_qubits': 64,
        'shots': 1024,
        'is_tensor_network': True,
        'is_stabilizer_hybrid': False,
        'is_binary_decision_tree': False,
        'long_range_columns': -1,
        'reverse_row_and_col': False,
        'sdrp': 0.045,
        'noise_model_short': 0.25,
        'noise_model_long': 0.25,
    }

    DEFAULT_CONFIGURATION = {
        'backend_name': 'ace_qasm_simulator',
        'backend_version': __version__,
        'n_qubits': 64,
        'conditional': True,
        'url': 'https://github.com/vm6502q/qiskit-qrack-provider',
        'simulator': True,
        'local': True,
        'open_pulse': False,
        'memory': False,
        'max_memory_mb' :None,
        'description': 'A fundamentally-optimized, approximate, nearest-neighbor qasm simulator',
        'coupling_map': None,
        'noise_model': None,
        'basis_gates': [
            'id', 'u', 'u1', 'u2', 'u3', 'r', 'rx', 'ry', 'rz',
            'h', 'x', 'y', 'z', 's', 'sdg', 'sx', 'sxdg', 'p', 't', 'tdg',
            'cx', 'cy', 'cz', 'swap', 'iswap', 'reset', 'measure'
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
            'name': 'dcx',
            'parameters': [],
            'conditional': True,
            'description': 'Double-CNOT gate',
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
        }]
    }

    def _factor_width(self, width, reverse=False):
        col_len = math.floor(math.sqrt(width))
        while ((width // col_len) * col_len) != width:
            col_len -= 1
        row_len = width // col_len

        self._col_length = row_len if reverse else col_len
        self._row_length = col_len if reverse else row_len

    # Mostly written by Dan, but with a little help from Elara (custom OpenAI GPT)
    def get_logical_coupling_map(self):
        if self._coupling_map:
            return self._coupling_map

        coupling_map = set()
        rows, cols = self._row_length, self._col_length

        # Map each column index to its full list of logical qubit indices
        def logical_index(row, col):
            return row * cols + col

        for col in range(cols):
            connected_cols = [col]
            c = (col - 1) % cols
            while self._is_col_long_range[c] and (len(connected_cols) < self._row_length):
                connected_cols.append(c)
                c = (c - 1) % cols
            if len(connected_cols) < self._row_length:
                connected_cols.append(c)
            c = (col + 1) % cols
            while self._is_col_long_range[c] and (len(connected_cols) < self._row_length):
                connected_cols.append(c)
                c = (c + 1) % cols
            if len(connected_cols) < self._row_length:
                connected_cols.append(c)

            for row in range(rows):
                a = logical_index(row, col)
                for c in connected_cols:
                    for r in range(0, rows):
                        b = logical_index(r, c)
                        if a != b:
                            coupling_map.add((a, b))

        return sorted(coupling_map)

    # Designed by Dan, and implemented by Elara:
    def get_noise_model(self):
        if self._noise_model:
            return self._noise_model

        noise_model = NoiseModel()
        x, y = self._options['noise_model_short'], self._options['noise_model_long']

        for a, b in self.get_logical_coupling_map():
            col_a, col_b = a % self._row_length, b % self._row_length
            row_a, row_b = a // self._row_length, b // self._row_length
            is_long_a = self._is_col_long_range[col_a]
            is_long_b = self._is_col_long_range[col_b]

            if is_long_a and is_long_b:
                continue  # No noise on long-to-long

            same_col = col_a == col_b

            if same_col:
                continue  # No noise for same column

            if is_long_a or is_long_b:
                y_cy = 1 - (1 - y) ** 2
                y_swap = 1 - (1 - y) ** 3
                noise_model.add_quantum_error(depolarizing_error(y, 2), "cx", [a, b])
                noise_model.add_quantum_error(depolarizing_error(y_cy, 2), "cy", [a, b])
                noise_model.add_quantum_error(depolarizing_error(y_cy, 2), "cz", [a, b])
                noise_model.add_quantum_error(
                    depolarizing_error(y_swap, 2), "swap", [a, b]
                )
            else:
                y_cy = 1 - (1 - y) ** 2
                y_swap = 1 - (1 - y) ** 3
                noise_model.add_quantum_error(depolarizing_error(y_cy, 2), "cx", [a, b])
                noise_model.add_quantum_error(depolarizing_error(y_cy, 2), "cy", [a, b])
                noise_model.add_quantum_error(depolarizing_error(y_cy, 2), "cz", [a, b])
                noise_model.add_quantum_error(
                    depolarizing_error(y_swap, 2), "swap", [a, b]
                )

        return noise_model

    max_circuits = None
    @property
    def target(self):
        if self._target is None:
            self._target = Target.from_configuration(
                basis_gates=self._configuration['basis_gates'],
                num_qubits=self._number_of_qubits,
                coupling_map=None,
                instruction_durations=None,
                concurrent_measurements=None,
                dt=None,
                timing_constraints=None,
                custom_name_mapping=None
            )
        return self._target

    def max_memory_mb(self):
        return None

    def __init__(self, configuration=None, provider=None, **fields):
        """Initialize a backend class

        Args:
            configuration (dict): A backend configuration
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

        configuration = configuration or self.DEFAULT_CONFIGURATION

        self._number_of_clbits = 0
        self._shots = 1
        self._coupling_map = None
        self._noise_model = None
        self._target = None
        self._options = self._default_options()
        self._provider = provider
        if fields:
            for field in fields:
                if  field not in self.DEFAULT_OPTIONS:
                    raise AttributeError("Options field %s is not valid for this backend" % field)
            self._options.update_options(**fields)

        self._number_of_qubits = self._options['n_qubits'] if 'n_qubits' in self._options else configuration['n_qubits']
        self._factor_width(self._number_of_qubits, self._options['reverse_row_and_col'])

        long_range_columns = self._options['long_range_columns'] if 'long_range_columns' in self._options else DEFAULT_OPTIONS['long_range_columns']
        if long_range_columns < 0:
            long_range_columns = 3 if (self._row_length % 3) == 1 else 2
        col_seq = [True] * long_range_columns + [False]
        len_col_seq = len(col_seq)
        self._is_col_long_range = (
            col_seq * ((self._row_length + len_col_seq - 1) // len_col_seq)
        )[: self._row_length]
        if long_range_columns < self._row_length:
            self._is_col_long_range[-1] = False

        if configuration['coupling_map'] is None:
            self._coupling_map = self.get_logical_coupling_map()
            configuration['coupling_map'] = CouplingMap(self._coupling_map)

        if configuration['noise_model'] is None:
            self._noise_model = self.get_noise_model()
            configuration['noise_model'] = self._noise_model

        self._target = Target()
        single_target_dict = {(q,): None for q in range(self._number_of_qubits)}
        self._target.add_instruction(IGate(), single_target_dict)
        self._target.add_instruction(U3Gate(Parameter('theta'), Parameter('phi'), Parameter('lambda')), single_target_dict)
        self._target.add_instruction(U2Gate(Parameter('phi'), Parameter('lambda')), single_target_dict)
        self._target.add_instruction(U1Gate(Parameter('theta')), single_target_dict)
        self._target.add_instruction(XGate(), single_target_dict)
        self._target.add_instruction(YGate(), single_target_dict)
        self._target.add_instruction(ZGate(), single_target_dict)
        self._target.add_instruction(HGate(), single_target_dict)
        self._target.add_instruction(RXGate(Parameter('theta')), single_target_dict)
        self._target.add_instruction(RYGate(Parameter('theta')), single_target_dict)
        self._target.add_instruction(RZGate(Parameter('theta')), single_target_dict)
        self._target.add_instruction(SGate(), single_target_dict)
        self._target.add_instruction(SdgGate(), single_target_dict)
        self._target.add_instruction(TGate(), single_target_dict)
        self._target.add_instruction(TdgGate(), single_target_dict)

        double_target_dict = {(q1, q2,): None for q1, q2 in self._coupling_map}
        self._target.add_instruction(CXGate(), double_target_dict)
        self._target.add_instruction(CYGate(), double_target_dict)
        self._target.add_instruction(CZGate(), double_target_dict)
        self._target.add_instruction(SwapGate(), double_target_dict)
        self._target.add_instruction(iSwapGate(), double_target_dict)

        self._coupling_map = configuration['coupling_map']

        self._configuration = configuration

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
        # WARNING: The above prototype for return type doesn't work in BackendV2 in Qiskit v0.30.0.
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
            'isStabilizerHybrid': options.is_stabilizer_hybrid if hasattr(options, 'is_stabilizer_hybrid') else self._options.get('is_stabilizer_hybrid'),
            'isBinaryDecisionTree': options.is_binary_decision_tree if hasattr(options, 'is_binary_decision_tree') else self._options.get('is_binary_decision_tree'),
            'long_range_columns': options.long_range_columns if hasattr(options, 'long_range_columns') else self._options.get('long_range_columns'),
            'reverse_row_and_col': options.reverse_row_and_col if hasattr(options, 'reverse_row_and_col') else self._options.get('reverse_row_and_col'),
        }

        data = run_input.config.memory if hasattr(run_input, 'config') else []
        self._shots = options['shots'] if 'shots' in options else (run_input.config.shots if hasattr(run_input, 'config') else self._options.get('shots'))
        self._sdrp = options['sdrp'] if 'sdrp' in options else (run_input.config.sdrp if hasattr(run_input, 'config') else self._options.get('sdrp'))
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
            backend_version = self._configuration['backend_version'],
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
            experiment (QuantumCircuit): experiment from qobj experiments list
        Returns:
            dict: A dictionary of results.
            dict: A result dictionary
        Raises:
            QrackError: If the number of qubits is too large, or another
                error occurs during execution.
        """
        start = time.time()

        instructions = []
        if isinstance(experiment, QuantumCircuit):
            self._number_of_qubits = len(experiment.qubits)
            self._number_of_clbits = len(experiment.clbits)
            for datum in experiment.data:
                qubits = []
                for qubit in datum[1]:
                    qubits.append(experiment.qubits.index(qubit))

                clbits = []
                for clbit in datum[2]:
                    clbits.append(experiment.clbits.index(clbit))

                conditional = None
                condition = getattr(datum[0], "condition", None)
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

                instructions.append({
                    'name': datum[0].name,
                    'qubits': qubits,
                    'memory': clbits,
                    'condition': condition,
                    'conditional': conditional,
                    'params': datum[0].params
                })
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

        if ('noise' in options) and (options['noise'] > 0):
            boundary_start = 1
            shotsPerLoop = 1
            shotLoopMax = self._shots
            self._sample_measure = False
        else:
            for opcount in range(len(instructions)):
                operation = instructions[opcount]

                if (operation['name'] == 'id') or (operation['name'] == 'barrier'):
                    continue

                if is_initializing and ((operation['name'] == 'measure') or (operation['name'] == 'reset')):
                    continue

                is_initializing = False

                if (operation['name'] == 'measure') or (operation['name'] == 'reset'):
                    if boundary_start == -1:
                        boundary_start = opcount

                if (boundary_start != -1) and (operation['name'] != 'measure'):
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
                self._sim = QrackAceBackend(self._number_of_qubits, **options)
                self._sim.sim.set_sdrp(self._sdrp)
                self._classical_memory = 0
                self._classical_register = 0

                for operation in instructions[:boundary_start]:
                    self._apply_op(operation)

                preamble_memory = self._classical_memory
                preamble_register = self._classical_register
                preamble_sim = self._sim

        for shot in range(shotLoopMax):
            if preamble_sim is None:
                self._sim = QrackAceBackend(self._number_of_qubits, **options)
                self._sim.sim.set_sdrp(self._sdrp)
                self._classical_memory = 0
                self._classical_register = 0
            else:
                self._sim = QrackAceBackend(toClone = preamble_sim)
                self._sim.sim.set_sdrp(self._sdrp)
                self._classical_memory = preamble_memory
                self._classical_register = preamble_register

            for operation in instructions[boundary_start:]:
                self._apply_op(operation)

            if not self._sample_measure and (len(self._sample_qubits) > 0):
                self._data += [bin(self._classical_memory)[2:].zfill(self._number_of_qubits)]
                self._sample_qubits = []
                self._sample_clbits = []
                self._sample_cregbits = []

        if self._sample_measure and (len(self._sample_qubits) > 0):
            self._data = self._add_sample_measure(self._sample_qubits, self._sample_clbits, self._shots)

        data = pd.DataFrame(data={ 'counts': dict(Counter(self._data)) })

        metadata = { 'measure_sampling': self._sample_measure }
        if isinstance(experiment, QuantumCircuit) and hasattr(experiment, 'metadata') and experiment.metadata:
            metadata = experiment.metadata
            metadata['measure_sampling'] = self._sample_measure

        return QrackExperimentResult(
            shots = self._shots,
            data = data,
            status = 'DONE',
            success = True,
            header = QrackExperimentResultHeader(name = experiment.name),
            metadata = metadata,
            time_taken = (time.time() - start)
        )

    def _apply_op(self, operation):
        name = operation.name

        if (name == "id") or (name == "barrier"):
            # Skip measurement logic
            return

        conditional = getattr(operation, "conditional", None)
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

        if (name == "u1") or (name == "p"):
            self._sim.u(operation.qubits[0]._index, 0, 0, float(operation.params[0]))
        elif name == "u2":
            self._sim.u(
                operation.qubits[0]._index,
                math.pi / 2,
                float(operation.params[0]),
                float(operation.params[1]),
            )
        elif (name == "u3") or (name == "u"):
            self._sim.u(
                operation.qubits[0]._index,
                float(operation.params[0]),
                float(operation.params[1]),
                float(operation.params[2]),
            )
        elif name == "r":
            self._sim.u(
                operation.qubits[0]._index,
                float(operation.params[0]),
                float(operation.params[1]) - math.pi / 2,
                (-1 * float(operation.params[1])) + math.pi / 2,
            )
        elif name == "rx":
            self._sim.r(
                Pauli.PauliX, float(operation.params[0]), operation.qubits[0]._index
            )
        elif name == "ry":
            self._sim.r(
                Pauli.PauliY, float(operation.params[0]), operation.qubits[0]._index
            )
        elif name == "rz":
            self._sim.r(
                Pauli.PauliZ, float(operation.params[0]), operation.qubits[0]._index
            )
        elif name == "h":
            self._sim.h(operation.qubits[0]._index)
        elif name == "x":
            self._sim.x(operation.qubits[0]._index)
        elif name == "y":
            self._sim.y(operation.qubits[0]._index)
        elif name == "z":
            self._sim.z(operation.qubits[0]._index)
        elif name == "s":
            self._sim.s(operation.qubits[0]._index)
        elif name == "sdg":
            self._sim.adjs(operation.qubits[0]._index)
        elif name == "t":
            self._sim.t(operation.qubits[0]._index)
        elif name == "tdg":
            self._sim.adjt(operation.qubits[0]._index)
        elif name == "cx":
            self._sim.cx(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == "cy":
            self._sim.cy(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == "cz":
            self._sim.cz(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == "dcx":
            self._sim.mcx(operation.qubits[0]._index, operation.qubits[1]._index)
            self._sim.mcx(operation.qubits[1]._index, operation.qubits[0]._index)
        elif name == "swap":
            self._sim.swap(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == "iswap":
            self._sim.iswap(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == "iswap_dg":
            self._sim.adjiswap(operation.qubits[0]._index, operation.qubits[1]._index)
        elif name == "reset":
            qubits = operation.qubits
            for qubit in qubits:
                if self._sim.m(qubit._index):
                    self._sim.x(qubit._index)
        elif name == "measure":
            qubits = operation.qubits
            clbits = operation.clbits
            cregbits = (
                operation.register
                if hasattr(operation, "register")
                else len(operation.qubits) * [-1]
            )

            self._sample_qubits += qubits
            self._sample_clbits += clbits
            self._sample_cregbits += cregbits

            if not self._sample_measure:
                for index in range(len(qubits)):
                    qubit_outcome = self._sim.m(qubits[index]._index)

                    clbit = clbits[index]
                    clmask = 1 << clbit
                    self._classical_memory = (self._classical_memory & (~clmask)) | (
                        qubit_outcome << clbit
                    )

                    cregbit = cregbits[index]
                    if cregbit < 0:
                        cregbit = clbit

                    regbit = 1 << cregbit
                    self._classical_register = (
                        self._classical_register & (~regbit)
                    ) | (qubit_outcome << cregbit)

        elif name == "bfunc":
            mask = int(operation.mask, 16)
            relation = operation.relation
            val = int(operation.val, 16)

            cregbit = operation.register
            cmembit = operation.memory if hasattr(operation, "memory") else None

            compared = (self._classical_register & mask) - val

            if relation == "==":
                outcome = compared == 0
            elif relation == "!=":
                outcome = compared != 0
            elif relation == "<":
                outcome = compared < 0
            elif relation == "<=":
                outcome = compared <= 0
            elif relation == ">":
                outcome = compared > 0
            elif relation == ">=":
                outcome = compared >= 0
            else:
                raise QrackError("Invalid boolean function relation.")

            # Store outcome in register and optionally memory slot
            regbit = 1 << cregbit
            self._classical_register = (self._classical_register & (~regbit)) | (
                int(outcome) << cregbit
            )
            if cmembit is not None:
                membit = 1 << cmembit
                self._classical_memory = (self._classical_memory & (~membit)) | (
                    int(outcome) << cmembit
                )
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

            data.append(bin(self._classical_memory)[2:].zfill(self._number_of_qubits))

        return data

    @staticmethod
    def name():
        return 'qasm_simulator'
