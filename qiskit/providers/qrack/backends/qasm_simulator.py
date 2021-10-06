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
from qiskit.providers.options import Options
from qiskit.result import Result

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.qobj.qasm_qobj import QasmQobjExperiment, QasmQobjInstruction

class QrackExperimentResultHeader:
    def __init__(self, name):
        self.name = name


class QrackExperimentResult:
    def __init__(self, shots, data, status, success, header, meta_data = None, time_taken = None):
        self.shots = shots
        self.data = data
        self.status = status
        self.success = success
        self.header = header
        self.meta_data = meta_data,
        self.time_taken = time_taken


class QasmSimulator(BackendV1):
    """
    Contains an OpenCL based backend
    """

    DEFAULT_OPTIONS = {
        'method': 'automatic',
        'is_multi_device': True,
        'is_schmidt_decompose': True,
        'is_stabilizer_hybrid': True,
        'is_1qb_fusion': True,
        'is_cpu_gpu_hybrid': True
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
            'u1', 'u2', 'u3', 'u', 'p', 'r', 'cx', 'cz', 'ch', 'id', 'x', 'sx', 'y', 'z', 'h',
            'rx', 'ry', 'rz', 's', 'sdg', 't', 'tdg', 'iswap', 'swap', 'ccx', 'initialize', 'cu1', 'cu2',
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
            'parameters': ['theta', 'phi'],
            'conditional': True,
            'description': 'Single-qubit gate [[cos(theta), -1j*exp(-1j*phi)], [sin(theta), -1j*exp(1j *phi)*sin(theta), cos(theta)]]',
            'qasm_def': 'gate r(theta, phi) q { U(theta, phi - pi/2, -phi + pi/2) q;}'
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
            'name': 'iswap',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit ISWAP gate',
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
        }, {
            'name': 'reset',
            'parameters': [],
            'conditional': True,
            'description': 'Reset qubit to 0 state',
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
        self._number_of_cbits = 0
        self._shots = 1
        self._memory = 0

        self._options = self._default_options()

        super().__init__(configuration=configuration, provider=provider, **fields)

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
            'isMultiDevice': options.is_multi_device if hasattr(options, 'is_multi_device') else self._options.get('is_multi_device'),
            'isSchmidtDecompose': options.is_schmidt_decompose if hasattr(options, 'is_schmidt_decompose') else self._options.get('is_schmidt_decompose'),
            'isStabilizerHybrid': options.is_stabilizer_hybrid if hasattr(options, 'is_stabilizer_hybrid') else self._options.get('is_stabilizer_hybrid'),
            'is1QbFusion': options.is_1qb_fusion if hasattr(options, 'is_1qb_fusion') else self._options.get('is_1qb_fusion'),
            'isCpuGpuHybrid': options.is_cpu_gpu_hybrid if hasattr(options, 'is_cpu_gpu_hybrid') else self._options.get('is_cpu_gpu_hybrid')
        }

        job_id = str(uuid.uuid4())
        job = QrackJob(self, job_id, self._run_job(job_id, run_input, **qrack_options), run_input)
        return job

    def _run_job(self, job_id, run_input, **options):
        """Run experiments in run_input
        Args:
            job_id (str): unique id for the job.
            run_input (QuantumCircuit or Schedule or list): job description
        Returns:
            Result: Result object
        """
        if hasattr(run_input, 'config'):
            self._shots = run_input.config.shots
            self._memory = run_input.config.memory
        else:
            self._shots = 1
            self._memory = 0

        experiments = run_input.experiments if hasattr(run_input, 'config') else run_input
        results = []

        start = time.time()
        for experiment in experiments:
            results.append(self.run_experiment(experiment, **options))
        end = time.time()

        return Result(
            backend_name = self.name(),
            backend_version = self._configuration.backend_version,
            qobj_id = run_input.qobj_id if hasattr(run_input, 'config') else '',
            job_id = job_id,
            success = True,
            results = results,
            date = datetime.now(),
            status = 'COMPLETED',
            header = run_input.header.to_dict() if hasattr(run_input, 'config') else {},
            time_taken = (end - start)
        )

        return Result.from_dict(result)

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
        instructions = []
        if isinstance(experiment, QasmQobjExperiment):
            self._number_of_qubits = experiment.header.n_qubits
            self._number_of_cbits = experiment.header.memory_slots
            instructions = experiment.instructions
        elif isinstance(experiment, QuantumCircuit):
            self._number_of_qubits = len(experiment.qubits)
            self._number_of_cbits = len(experiment.clbits)
            for datum in experiment._data:

                qubits = []
                for qubit in datum[1]:
                    qubits.append(experiment.qubits.index(qubit))

                clbits = []
                for clbit in datum[2]:
                    clbits.append(experiment.clbits.index(clbit))

                instructions.append(QasmQobjInstruction(datum[0].name, qubits = qubits, memory = clbits, params = datum[0].params))
        else:
            raise QrackError('Unrecognized "run_input" argument specified for run().')
        self._sample_qubits = []
        self._sample_clbits = []
        self._sample_cregbits = []
        self.__memory = []
        self._sample_measure = True

        start = time.time()

        shotsPerLoop = self._shots
        shotLoopMax = 1

        did_measure = False
        is_initializing = True
        opcount = -1
        nonunitary_start = 0

        if self._shots != 1:
            for operation in instructions:
                opcount = opcount + 1

                if operation.name == 'id' or operation.name == 'barrier':
                    continue

                if hasattr(operation, 'conditional') or operation.name == 'reset':
                    if is_initializing:
                        continue
                    if operation.name != 'reset':
                        nonunitary_start = opcount
                    shotLoopMax = self._shots
                    shotsPerLoop = 1
                    self._sample_measure = False
                    break

                is_initializing = False

                if operation.name == 'measure':
                    if nonunitary_start == 0:
                        nonunitary_start = opcount
                    did_measure = True
                elif did_measure:
                    shotLoopMax = self._shots
                    shotsPerLoop = 1
                    self._sample_measure = False
                    break

        preamble_classical_memory = 0
        preamble_classical_register = 0

        is_unitary_preamble = False

        if self._sample_measure:
            nonunitary_start = 0
        else:
            is_unitary_preamble = True
            self._sample_measure = True
            self._sim = QrackSimulator(self._number_of_qubits, **options)
            self._classical_memory = 0
            self._classical_register = 0

            for operation in instructions[:nonunitary_start]:
                self._apply_op(operation, shotsPerLoop)

            preamble_classical_memory = self._classical_memory
            preamble_classical_register = self._classical_register
            self._sample_measure = False

        preamble_sim = self._sim if is_unitary_preamble else None

        for shot in range(shotLoopMax):
            if not is_unitary_preamble:
                self._sim = QrackSimulator(qubitCount = self._number_of_qubits, **options)
                self._classical_memory = 0
                self._classical_register = 0
            else:
                self._sim = QrackSimulator(cloneSid = preamble_sim.sid, **options)
                self._classical_memory = preamble_classical_memory
                self._classical_register = preamble_classical_register

            for operation in instructions[nonunitary_start:]:
                self._apply_op(operation, shotsPerLoop)

            if len(self._sample_qubits) > 0:
                if self._sample_measure:
                    self.__memory = self._add_sample_measure(self._sample_qubits, self._sample_clbits, shotsPerLoop)
                else:
                    self._add_qasm_measure(self._sample_qubits, self._sample_clbits, self._sample_cregbits)
                self._sample_qubits = []
                self._sample_clbits = []
                self._sample_cregbits = []

            if not self._sample_measure:
                # Turn classical_memory (int) into bit string and pad zero for unused cmembits
                self.__memory.append(hex(int(bin(self._classical_memory)[2:], 2)))

        end = time.time()

        data = { 'counts': dict(Counter(self.__memory)) }
        dfData = pd.DataFrame(data=data)
        data['memory'] = self.__memory

        if isinstance(experiment, QasmQobjExperiment):
            return {
                'name': experiment.header.name,
                'shots': self._shots,
                'data': data,
                'status': 'DONE',
                'success': True,
                'time_taken': (end - start),
                'header': experiment.header.to_dict(),
                'metadata': { 'measure_sampling' : self._sample_measure }
            }

        return QrackExperimentResult(
            shots = self._shots,
            data = dfData,
            status = 'DONE',
            success = True,
            header = QrackExperimentResultHeader(name = experiment.name),
            meta_data = { 'measure_sampling' : self._sample_measure },
            time_taken = (end - start)
        )

    #@profile
    def _apply_op(self, operation, shotsPerLoop):
        name = operation.name

        if name == 'id':
            # Skip measurement logic
            return
        elif name == 'barrier':
            # Skip measurement logic
            return

        if (name != 'measure' or hasattr(operation, 'conditional')) and len(self._sample_qubits) > 0:
            if self._sample_measure:
                self._memory = self._add_sample_measure(self._sample_qubits, sample_clbits, sim, shotsPerLoop)
            else:
                self._add_qasm_measure(self._sample_qubits, self._sample_clbits, self._sample_cregbits)
            self._sample_qubits = []
            self._sample_clbits = []
            self._sample_cregbits = []

        conditional = getattr(operation, 'conditional', None)
        if isinstance(conditional, int):
            conditional_bit_set = (self._classical_register >> conditional) & 1
            if not conditional_bit_set:
                return
        elif conditional is not None:
            mask = int(operation.conditional.mask, 16)
            if mask > 0:
                value = self._classical_memory & mask
                while (mask & 0x1) == 0:
                    mask >>= 1
                    value >>= 1
                if value != int(operation.conditional.val, 16):
                    return

        if (name == 'u1') or (name == 'p'):
            self._sim.u(operation.qubits[0], 0, 0, operation.params[0])
        elif name == 'u2':
            self._sim.u(operation.qubits[0], np.pi / 2, operation.params[0], operation.params[1])
        elif (name == 'u3') or (name == 'u'):
            self._sim.u(operation.qubits[0], operation.params[0], operation.params[1], operation.params[2])
        elif name == 'r':
            self._sim.u(operation.qubits[0], operation.params[0], operation.params[1] - np.pi/2, -operation.params[1] + np.pi/2)
        elif name == 'cx':
            self._sim.mcx(operation.qubits[0:1], operation.qubits[1])
        elif name == 'cy':
            self._sim.mcy(operation.qubits[0:1], operation.qubits[1])
        elif name == 'cz':
            self._sim.mcz(operation.qubits[0:1], operation.qubits[1])
        elif name == 'ch':
            self._sim.mch(operation.qubits[0:1], operation.qubits[1])
        elif name == 'x':
            self._sim.x(operation.qubits[0])
        elif name == 'y':
            self._sim.y(operation.qubits[0])
        elif name == 'z':
            self._sim.z(operation.qubits[0])
        elif name == 'h':
            self._sim.h(operation.qubits[0])
        elif name == 'rx':
            self._sim.r(Pauli.PauliX, operation.params[0], operation.qubits[0])
        elif name == 'ry':
            self._sim.r(Pauli.PauliY, operation.params[0], operation.qubits[0])
        elif name == 'rz':
            self._sim.r(Pauli.PauliZ, operation.params[0], operation.qubits[0])
        elif name == 's':
            self._sim.s(operation.qubits[0])
        elif name == 'sdg':
            self._sim.adjs(operation.qubits[0])
        elif name == 't':
            self._sim.t(operation.qubits[0])
        elif name == 'tdg':
            self._sim.adjt(operation.qubits[0])
        elif name == 'swap':
            self._sim.swap(operation.qubits[0], operation.qubits[1])
        elif name == 'iswap':
            self._sim.iswap(operation.qubits[0], operation.qubits[1])
        elif name == 'ccx':
            self._sim.mcx(operation.qubits[0:2], operation.qubits[2])
        elif name == 'cu1':
            self._sim.mcu(operation.qubits[0:1], operation.qubits[1], 0, 0, operation.params[0])
        elif name == 'cu2':
            self._sim.mcu(operation.qubits[0:1], operation.qubits[1], np.pi / 2, operation.params[0], operation.params[1])
        elif name == 'cu3':
            self._sim.mcu(operation.qubits[0:1], operation.qubits[1], operation.params[0], operation.params[1], operation.params[2])
        elif name == 'cswap':
            self._sim.cswap(operation.qubits[0:1], operation.qubits[1], operation.qubits[2])
        elif name == 'mcx':
            self._sim.mcx(operation.qubits[0:-1], operation.qubits[-1])
        elif name == 'mcy':
            self._sim.mcy(operation.qubits[0:-1], operation.qubits[-1])
        elif name == 'mcz':
            self._sim.mcz(operation.qubits[0:-1], operation.qubits[-1])
        elif name == 'mcswap':
            self._sim.cswap(operation.qubits[:-2], operation.qubits[-2], operation.qubits[-1])
        elif name == 'reset':
            mres = self._sim.measure_pauli([Pauli.PauliZ] * len(operation.qubits), operation.qubits)
            for i in range(len(operation.qubits)):
                if ((mres >> i) & 1) > 0:
                    self._sim.x(operation.qubits[i])
        elif name == 'measure':
            cregbits = operation.register if hasattr(operation, 'register') else len(operation.qubits) * [-1]
            self._sample_qubits.append(operation.qubits)
            self._sample_clbits.append(operation.memory)
            self._sample_cregbits.append(cregbits)
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

    #@profile
    def _add_sample_measure(self, sample_qubits, sample_clbits, num_samples):
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
            sample = self._sim.measure_pauli([Pauli.PauliZ] * len(measure_qubit), measure_qubit)
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
        measure_results = self._sim.measure_shots(measure_qubit, num_samples)
        classical_state = self._classical_memory
        for key in measure_results:
            sample = key
            classical_state = self._classical_memory
            for index in range(len(measure_qubit)):
                qubit = measure_qubit[index]
                cbit = measure_clbit[index]
                qubit_outcome = (sample >> index) & 1
                bit = 1 << cbit
                classical_state = (classical_state & (~bit)) | (qubit_outcome << cbit)
            outKey = bin(classical_state)[2:]
            memory.append(hex(int(outKey, 2)))

        return memory

    #@profile
    def _add_qasm_measure(self, sample_qubits, sample_clbits, sample_cregbits=None):
        """Apply a measure instruction to a qubit.
        Args:
            qubit (int): qubit is the qubit measured.
            cmembit (int): is the classical memory bit to store outcome in.
            cregbit (int, optional): is the classical register bit to store outcome in.
        """

        measure_qubit = [qubit for sublist in sample_qubits for qubit in sublist]
        measure_clbit = [clbit for sublist in sample_clbits for clbit in sublist]
        measure_cregbit = [clbit for sublist in sample_cregbits for clbit in sublist]

        sample = self._sim.measure_pauli([Pauli.PauliZ] * len(measure_qubit), measure_qubit)
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

    @staticmethod
    def name():
        return 'qasm_simulator'
