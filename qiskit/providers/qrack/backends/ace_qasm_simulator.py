# This code is based on and adapted from https://github.com/Qiskit/qiskit-qcgpu-provider/blob/master/qiskit_qcgpu_provider/qasm_simulator.py
#
# Adapted by Daniel Strano. Many thanks to Adam Kelly for pioneering a third-party Qiskit provider.
# Anthropic Claude adapted this back-end wrapper for Qiskit v2.4.2.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Updated for Qiskit v2 (2.x) compatibility:
#   - BackendV2.__init__ signature: name/description as kwargs
#   - CircuitInstruction API: .operation / .qubits / .clbits (not tuple indexing)
#   - Target built via add_instruction + InstructionProperties (not from_configuration)
#   - Result: qobj_id removed
#   - name is a str attribute, not a @staticmethod
#   - ExperimentResult uses qiskit.result.models.ExperimentResult/ExperimentResultData

import uuid
import time
import math
import numpy as np
from datetime import datetime
from collections import Counter

from ..version import __version__
from ..qrackjob import QrackJob
from ..qrackerror import QrackError
from pyqrack import QrackAceBackend, Pauli

from qiskit.providers.backend import BackendV2
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.providers.options import Options
from qiskit.transpiler import Target, CouplingMap, InstructionProperties
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import Clbit, Parameter

from qiskit.circuit.library import (
    IGate, UGate, U3Gate, U2Gate, U1Gate,
    XGate, YGate, ZGate, HGate,
    RXGate, RYGate, RZGate,
    SGate, SdgGate, TGate, TdgGate,
    CXGate, CYGate, CZGate, SwapGate, iSwapGate,
    Measure, Reset
)

from qiskit_aer.noise import NoiseModel, depolarizing_error


class QrackQasmQobjInstructionConditional:
    def __init__(self, mask, val):
        self.mask = mask
        self.val = val

    def to_dict(self):
        return vars(self)


class AceQasmSimulator(BackendV2):
    """
    Contains a fundamentally-optimized, approximate, nearest-neighbor backend.
    Updated for Qiskit v2.
    """

    DEFAULT_OPTIONS = {
        # This isn't MPS, but Qiskit won't let us allocate many qubits otherwise.
        'method': 'matrix_product_state',
        'n_qubits': 64,
        'shots': 1024,
        'is_schmidt_decompose_multi': True,
        'is_stabilizer_hybrid': False,
        'is_binary_decision_tree': False,
        'is_near_clifford_cpu': False,
        'is_gpu': True,
        'is_host_pointer': False,
        'noise': 0,
        'sdrp': 0.03,
        'long_range_columns': 4,
        'long_range_rows': 4,
        'is_transpose': False,
        'noise_model_short': 0.0,
        'noise_model_long': 0.5,
        'history_window': 0,
        'is_torus': True,
    }

    # Gates supported by QrackAceBackend
    BASIS_GATES = [
        'id', 'u', 'rx', 'ry', 'rz',
        'h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg',
        'cx', 'cy', 'cz', 'swap', 'iswap', 'reset', 'measure'
    ]

    max_circuits = None

    def __init__(self, provider=None, **fields):
        """Initialize AceQasmSimulator backend (Qiskit v2 compatible).

        Args:
            provider: Optional provider object.
            fields: kwargs for option overrides.
        """
        # Validate fields against known options
        for field in fields:
            if field not in self.DEFAULT_OPTIONS:
                raise AttributeError(
                    "Options field %s is not valid for this backend" % field
                )

        # BackendV2.__init__ takes name/description/backend_version as kwargs
        super().__init__(
            provider=provider,
            name='ace_qasm_simulator',
            description='A fundamentally-optimized, approximate, nearest-neighbor qasm simulator',
            backend_version=__version__,
        )

        self._options = self._default_options()
        if fields:
            self._options.update_options(**fields)

        self._number_of_qubits = self._options.get('n_qubits')
        self._coupling_map = None
        self._noise_model = None
        self._target = None

        # Build coupling map and noise model from a dummy backend instance
        long_range_columns = self._options.get('long_range_columns')
        long_range_rows = self._options.get('long_range_rows')
        # noise_model_short = self._options.get('noise_model_short')
        noise_model_long = self._options.get('noise_model_long')
        history_window = self._options.get('history_window')
        is_torus = self._options.get('is_torus')

        dummy = QrackAceBackend(
            self._number_of_qubits,
            long_range_columns=long_range_columns,
            long_range_rows=long_range_rows,
            history_window=history_window,
            is_torus=is_torus,
        )
        self._coupling_map = dummy.get_logical_coupling_map()
        self._noise_model = dummy.create_noise_model(
            x=noise_model_long,
        )
        dummy = None

    @classmethod
    def _default_options(cls):
        opts = Options()
        opts.update_options(**cls.DEFAULT_OPTIONS)
        return opts

    def get_logical_coupling_map(self):
        return self._coupling_map

    def get_noise_model(self):
        return self._noise_model

    @property
    def coupling_map(self):
        return CouplingMap(self._coupling_map) if self._coupling_map else None

    @property
    def target(self):
        if self._target is not None:
            return self._target

        # Build a proper Target with InstructionProperties so the transpiler
        # understands the gate set, connectivity, and approximate error rates.
        n = self._number_of_qubits
        tgt = Target(num_qubits=n, description=self.description)

        # Parameters for parameterised gates
        theta = Parameter('theta')
        phi   = Parameter('phi')
        lam   = Parameter('lam')

        # --- single-qubit gates: all qubits ---
        all_qubits = {(q,): InstructionProperties() for q in range(n)}
        # boundary qubits get a small depolarising penalty hint
        boundary = set()
        if self._coupling_map:
            for a, b in self._coupling_map:
                boundary.add(a)
                boundary.add(b)

        def _1q_props(err=0.0):
            return {
                (q,): InstructionProperties(error=(err if q in boundary else 0.0))
                for q in range(n)
            }

        tgt.add_instruction(IGate(),                 _1q_props())
        tgt.add_instruction(UGate(theta, phi, lam),  _1q_props())
        tgt.add_instruction(U3Gate(theta, phi, lam), _1q_props())
        tgt.add_instruction(U2Gate(phi, lam),        _1q_props())
        tgt.add_instruction(U1Gate(lam),             _1q_props())
        tgt.add_instruction(RXGate(theta),           _1q_props())
        tgt.add_instruction(RYGate(theta),           _1q_props())
        tgt.add_instruction(RZGate(theta),           _1q_props())
        tgt.add_instruction(HGate(),                 _1q_props())
        tgt.add_instruction(XGate(),                 _1q_props())
        tgt.add_instruction(YGate(),                 _1q_props())
        tgt.add_instruction(ZGate(),                 _1q_props())
        tgt.add_instruction(SGate(),                 _1q_props())
        tgt.add_instruction(SdgGate(),               _1q_props())
        tgt.add_instruction(TGate(),                 _1q_props())
        tgt.add_instruction(TdgGate(),               _1q_props())

        # --- two-qubit gates: coupled pairs ---
        e2 = self._options.get('noise_model_long') or 0.0
        if self._coupling_map:
            pair_props = {
                (a, b): InstructionProperties(error=e2)
                for a, b in self._coupling_map
            }
        else:
            # fully connected fallback
            pair_props = {
                (a, b): InstructionProperties(error=e2)
                for a in range(n) for b in range(n) if a != b
            }

        tgt.add_instruction(CXGate(),   pair_props)
        tgt.add_instruction(CYGate(),   pair_props)
        tgt.add_instruction(CZGate(),   pair_props)
        tgt.add_instruction(SwapGate(), pair_props)
        tgt.add_instruction(iSwapGate(), pair_props)

        # --- measure / reset: all qubits ---
        tgt.add_instruction(Measure(), {(q,): InstructionProperties() for q in range(n)})
        tgt.add_instruction(Reset(),   {(q,): InstructionProperties() for q in range(n)})

        self._target = tgt
        return self._target

    def run(self, run_input, **options):
        """Run a QuantumCircuit (or list) on this backend."""
        # Merge run-time options over defaults
        opts = dict(self._options._fields)  # copy defaults
        opts.update(options)

        qrack_options = {
            'is_schmidt_decompose_multi': opts.get('is_schmidt_decompose_multi'),
            'is_stabilizer_hybrid':       opts.get('is_stabilizer_hybrid'),
            'is_binary_decision_tree':    opts.get('is_binary_decision_tree'),
            'is_gpu':                     opts.get('is_gpu'),
            'is_near_clifford_tableau_writer': opts.get('is_near_clifford_cpu'),
            'is_host_pointer':            opts.get('is_host_pointer'),
            'noise':                      opts.get('noise'),
            'long_range_columns':         opts.get('long_range_columns'),
            'long_range_rows':            opts.get('long_range_rows'),
            'is_transpose':               opts.get('is_transpose'),
        }

        self._shots = opts.get('shots', 1024)
        self._sdrp  = opts.get('sdrp', 0.03)

        job_id = str(uuid.uuid4())
        job = QrackJob(
            self, job_id,
            self._run_job(job_id, run_input, **qrack_options),
            run_input,
        )
        return job

    def _run_job(self, job_id, run_input, **options):
        start = time.time()

        if isinstance(run_input, QuantumCircuit):
            experiments = [run_input]
        elif isinstance(run_input, list):
            experiments = run_input
        else:
            raise QrackError('Unrecognized run_input type: %s' % type(run_input))

        results = []
        for experiment in experiments:
            results.append(self._run_experiment(experiment, **options))

        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            job_id=job_id,
            success=True,
            results=results,
            date=datetime.now(),
            status='COMPLETED',
            header={},
        )

    def _run_experiment(self, experiment, **options):
        """Run a single QuantumCircuit and return an ExperimentResult."""
        start = time.time()

        if not isinstance(experiment, QuantumCircuit):
            raise QrackError('run_input must be a QuantumCircuit.')

        self._number_of_qubits  = len(experiment.qubits)
        self._number_of_clbits  = len(experiment.clbits)

        # --- Build instruction list using Qiskit v2 CircuitInstruction API ---
        instructions = []
        for datum in experiment.data:
            # Use .operation / .qubits / .clbits (tuple indexing is deprecated in v2)
            op     = datum.operation
            qubits = [experiment.find_bit(q).index for q in datum.qubits]
            clbits = [experiment.find_bit(c).index for c in datum.clbits]

            conditional = None
            condition = getattr(op, 'condition', None)
            if condition is not None:
                if isinstance(condition[0], Clbit):
                    conditional = experiment.find_bit(condition[0]).index
                else:
                    creg_index = experiment.cregs.index(condition[0])
                    size   = experiment.cregs[creg_index].size
                    offset = sum(len(experiment.cregs[i]) for i in range(creg_index))
                    mask   = ((1 << offset) - 1) ^ ((1 << (offset + size)) - 1)
                    val    = condition[1]
                    conditional = (
                        offset
                        if size == 1
                        else QrackQasmQobjInstructionConditional(mask, val)
                    )

            instructions.append({
                'name':        op.name,
                'qubits':      qubits,
                'memory':      clbits,
                'condition':   condition,
                'conditional': conditional,
                'params':      list(op.params),
            })

        # --- Determine shot / sample strategy ---
        self._sample_qubits   = []
        self._sample_clbits   = []
        self._sample_cregbits = []
        self._data            = []
        self._sample_measure  = True
        shotsPerLoop = self._shots
        shotLoopMax  = 1

        is_initializing = True
        boundary_start  = -1

        if options.get('noise', 0) > 0:
            boundary_start = 1
            shotsPerLoop   = 1
            shotLoopMax    = self._shots
            self._sample_measure = False
        else:
            for opcount, op in enumerate(instructions):
                name = op['name']
                if name in ('id', 'barrier'):
                    continue
                if is_initializing and name in ('measure', 'reset'):
                    continue
                is_initializing = False
                if name in ('measure', 'reset') and boundary_start == -1:
                    boundary_start = opcount
                if boundary_start != -1 and name != 'measure':
                    shotsPerLoop = 1
                    shotLoopMax  = self._shots
                    self._sample_measure = False
                    break

        if self._sample_measure or boundary_start <= 0:
            boundary_start = 0
            self._sample_measure = True
            shotsPerLoop = self._shots
            shotLoopMax  = 1

        # --- Preamble (instructions before first mid-circuit measurement) ---
        preamble_memory   = 0
        preamble_register = 0
        preamble_sim      = None

        if boundary_start > 1:
            self._sim = QrackAceBackend(self._number_of_qubits, **options)
            self._classical_memory   = 0
            self._classical_register = 0
            for op in instructions[:boundary_start]:
                self._apply_op(op)
            preamble_memory   = self._classical_memory
            preamble_register = self._classical_register
            preamble_sim      = self._sim

        # --- Shot loop ---
        for _ in range(shotLoopMax):
            if preamble_sim is None:
                self._sim = QrackAceBackend(self._number_of_qubits, **options)
                self._classical_memory   = 0
                self._classical_register = 0
            else:
                self._sim = preamble_sim.clone()
                self._classical_memory   = preamble_memory
                self._classical_register = preamble_register

            for op in instructions[boundary_start:]:
                self._apply_op(op)

            if not self._sample_measure and len(self._sample_qubits) > 0:
                self._data += [
                    bin(self._classical_memory)[2:].zfill(self._number_of_clbits)
                ]
                self._sample_qubits   = []
                self._sample_clbits   = []
                self._sample_cregbits = []

        if self._sample_measure and len(self._sample_qubits) > 0:
            self._data = self._add_sample_measure(
                self._sample_qubits, self._sample_clbits, self._shots
            )

        counts = dict(Counter(self._data))
        # Convert bit-string keys to hex format expected by Qiskit result
        hex_counts = {'0x%x' % int(k, 2): v for k, v in counts.items()}

        return ExperimentResult(
            shots=self._shots,
            success=True,
            data=ExperimentResultData(counts=hex_counts, memory=self._data),
            status='DONE',
            header={'name': experiment.name,
                    'n_qubits': self._number_of_qubits,
                    'creg_sizes': [[r.name, r.size] for r in experiment.cregs],
                    'memory_slots': self._number_of_clbits},
        )

    def _apply_op(self, operation):
        name = operation['name']

        if name in ('id', 'barrier'):
            return

        # Handle conditionals
        conditional = operation.get('conditional')
        if isinstance(conditional, int):
            if not ((self._classical_register >> conditional) & 1):
                return
        elif conditional is not None:
            mask = int(conditional.mask, 16)
            if mask > 0:
                value = self._classical_memory & mask
                while (mask & 0x1) == 0:
                    mask  >>= 1
                    value >>= 1
                if value != int(conditional.val, 16):
                    return

        qubits = operation['qubits']
        params = operation['params']

        if name in ('u1', 'p'):
            self._sim.u(qubits[0], 0, 0, float(params[0]))
        elif name == 'u2':
            self._sim.u(qubits[0], math.pi / 2, float(params[0]), float(params[1]))
        elif name in ('u3', 'u'):
            self._sim.u(qubits[0], float(params[0]), float(params[1]), float(params[2]))
        elif name == 'r':
            p1 = float(params[1])
            self._sim.u(qubits[0], float(params[0]), p1 - math.pi/2, -p1 + math.pi/2)
        elif name == 'rx':
            self._sim.r(Pauli.PauliX, float(params[0]), qubits[0])
        elif name == 'ry':
            self._sim.r(Pauli.PauliY, float(params[0]), qubits[0])
        elif name == 'rz':
            self._sim.r(Pauli.PauliZ, float(params[0]), qubits[0])
        elif name == 'h':
            self._sim.h(qubits[0])
        elif name == 'x':
            self._sim.x(qubits[0])
        elif name == 'y':
            self._sim.y(qubits[0])
        elif name == 'z':
            self._sim.z(qubits[0])
        elif name == 's':
            self._sim.s(qubits[0])
        elif name == 'sdg':
            self._sim.adjs(qubits[0])
        elif name == 't':
            self._sim.t(qubits[0])
        elif name == 'tdg':
            self._sim.adjt(qubits[0])
        elif name == 'cx':
            self._sim.cx(qubits[0], qubits[1])
        elif name == 'cy':
            self._sim.cy(qubits[0], qubits[1])
        elif name == 'cz':
            self._sim.cz(qubits[0], qubits[1])
        elif name == 'dcx':
            self._sim.cx(qubits[0], qubits[1])
            self._sim.cx(qubits[1], qubits[0])
        elif name == 'swap':
            self._sim.swap(qubits[0], qubits[1])
        elif name == 'iswap':
            self._sim.iswap(qubits[0], qubits[1])
        elif name in ('iswap_dg', 'adjiswap'):
            self._sim.adjiswap(qubits[0], qubits[1])
        elif name == 'reset':
            for q in qubits:
                if self._sim.m(q):
                    self._sim.x(q)
        elif name == 'measure':
            clbits = operation['memory']
            self._sample_qubits   += qubits
            self._sample_clbits   += clbits
            self._sample_cregbits += [-1] * len(qubits)

            if not self._sample_measure:
                for idx in range(len(qubits)):
                    outcome = self._sim.m(qubits[idx])
                    clbit   = clbits[idx]
                    clmask  = 1 << clbit
                    self._classical_memory = (
                        (self._classical_memory & ~clmask) | (outcome << clbit)
                    )
                    self._classical_register = (
                        (self._classical_register & ~clmask) | (outcome << clbit)
                    )
        elif name == 'bfunc':
            mask     = int(operation.get('mask', '0x0'), 16)
            relation = operation.get('relation', '==')
            val      = int(operation.get('val', '0x0'), 16)
            cregbit  = operation.get('register', 0)
            cmembit  = operation.get('memory_bit', None)
            compared = (self._classical_register & mask) - val
            outcome = {
                '==': compared == 0, '!=': compared != 0,
                '<':  compared <  0, '<=': compared <= 0,
                '>':  compared >  0, '>=': compared >= 0,
            }.get(relation)
            if outcome is None:
                raise QrackError("Invalid boolean function relation: %s" % relation)
            regbit = 1 << cregbit
            self._classical_register = (
                (self._classical_register & ~regbit) | (int(outcome) << cregbit)
            )
            if cmembit is not None:
                membit = 1 << cmembit
                self._classical_memory = (
                    (self._classical_memory & ~membit) | (int(outcome) << cmembit)
                )
        else:
            raise QrackError(
                '%s encountered unrecognized operation "%s"' % (self.name, name)
            )

    def _add_sample_measure(self, sample_qubits, sample_clbits, num_samples):
        measure_qubit = list(sample_qubits)
        measure_clbit = list(sample_clbits)

        if num_samples == 1:
            sample    = self._sim.m_all()
            result    = 0
            for idx in range(len(measure_qubit)):
                qubit_outcome = (sample >> measure_qubit[idx]) & 1
                result |= qubit_outcome << idx
            measure_results = [result]
        else:
            measure_results = self._sim.measure_shots(measure_qubit, num_samples)

        data = []
        for sample in measure_results:
            for idx in range(len(measure_qubit)):
                qubit_outcome = (sample >> idx) & 1
                clbit  = measure_clbit[idx]
                clmask = 1 << clbit
                self._classical_memory = (
                    (self._classical_memory & ~clmask) | (qubit_outcome << clbit)
                )
            data.append(bin(self._classical_memory)[2:].zfill(self._number_of_clbits))

        return data
