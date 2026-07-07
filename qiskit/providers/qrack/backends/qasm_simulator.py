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
#   - Target built via add_instruction + InstructionProperties (fully-connected, no noise)
#   - Result: qobj_id removed
#   - name is a str attribute, not a @staticmethod
#   - ExperimentResult uses qiskit.result.models.ExperimentResult/ExperimentResultData
#   - pandas dependency dropped
#   - inst_map/backend_properties removed (no longer in Qiskit v2 Target API)
#   - bfunc: consistent dict access for mask/relation/val
#   - operation.relation -> operation.get('relation') for consistency

import uuid
import time
import math
import numpy as np
from datetime import datetime
from collections import Counter

from ..version import __version__
from ..qrackjob import QrackJob
from ..qrackerror import QrackError
from pyqrack import QrackSimulator, Pauli

from qiskit.providers.backend import BackendV2
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.providers.options import Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import Clbit, Parameter

from qiskit.circuit.library import (
    IGate, UGate, U3Gate, U2Gate, U1Gate,
    XGate, YGate, ZGate, HGate, SGate, SdgGate, TGate, TdgGate,
    SXGate, SXdgGate,
    RXGate, RYGate, RZGate,
    CUGate, CXGate, CYGate, CZGate, CHGate,
    CPhaseGate, CSXGate,
    CCXGate, CCZGate,
    SwapGate, iSwapGate, CSwapGate,
    Measure, Reset,
)


class QrackQasmQobjInstructionConditional:
    def __init__(self, mask, val):
        self.mask = mask
        self.val = val

    def to_dict(self):
        return vars(self)


class QasmSimulator(BackendV2):
    """
    Contains an OpenCL based backend (Qiskit v2 compatible).
    Fully-connected topology; no noise model.
    """

    DEFAULT_OPTIONS = {
        'shots': 1024,
        'is_schmidt_decompose_multi': True,
        'is_stabilizer_hybrid': False,
        'is_binary_decision_tree': False,
        'is_approx_near_clifford': False,
        'is_near_clifford_cpu': False,
        'is_gpu': True,
        'is_host_pointer': False,
        'noise': 0,
        'sdrp': 0,
    }

    max_circuits = None

    def __init__(self, provider=None, **fields):
        """Initialize QasmSimulator (Qiskit v2).

        Args:
            provider: Optional provider object.
            fields: kwargs for option overrides.
        """
        for field in fields:
            if field not in self.DEFAULT_OPTIONS:
                raise AttributeError(
                    "Options field %s is not valid for this backend" % field
                )

        super().__init__(
            provider=provider,
            name='qasm_simulator',
            description='An OpenCL based qasm simulator',
            backend_version=__version__,
        )

        self._options = self._default_options()
        if fields:
            self._options.update_options(**fields)

        self._number_of_qubits = 64
        self._number_of_clbits = 0
        self._target = None

    @classmethod
    def _default_options(cls):
        opts = Options()
        opts.update_options(**cls.DEFAULT_OPTIONS)
        return opts

    @property
    def target(self):
        if self._target is not None:
            return self._target

        # Fully-connected target — no coupling map, no noise.
        # We build with a conservative qubit count; the transpiler will
        # see all pairs as connected since we pass None coupling_map.
        n = self._number_of_qubits
        tgt = Target(num_qubits=n, description=self.description)

        theta = Parameter('theta')
        phi   = Parameter('phi')
        lam   = Parameter('lam')
        gam   = Parameter('gamma')

        def _all_1q():
            return {(q,): InstructionProperties() for q in range(n)}

        def _all_2q():
            return {
                (a, b): InstructionProperties()
                for a in range(n) for b in range(n) if a != b
            }

        def _all_3q():
            return {
                (a, b, c): InstructionProperties()
                for a in range(n) for b in range(n) for c in range(n)
                if len({a, b, c}) == 3
            }

        # Single-qubit
        tgt.add_instruction(IGate(),               _all_1q())
        tgt.add_instruction(UGate(theta, phi, lam), _all_1q())
        tgt.add_instruction(U3Gate(theta, phi, lam), _all_1q())
        tgt.add_instruction(U2Gate(phi, lam),      _all_1q())
        tgt.add_instruction(U1Gate(lam),           _all_1q())
        tgt.add_instruction(RXGate(theta),         _all_1q())
        tgt.add_instruction(RYGate(theta),         _all_1q())
        tgt.add_instruction(RZGate(theta),         _all_1q())
        tgt.add_instruction(HGate(),               _all_1q())
        tgt.add_instruction(XGate(),               _all_1q())
        tgt.add_instruction(YGate(),               _all_1q())
        tgt.add_instruction(ZGate(),               _all_1q())
        tgt.add_instruction(SGate(),               _all_1q())
        tgt.add_instruction(SdgGate(),             _all_1q())
        tgt.add_instruction(SXGate(),              _all_1q())
        tgt.add_instruction(SXdgGate(),            _all_1q())
        tgt.add_instruction(TGate(),               _all_1q())
        tgt.add_instruction(TdgGate(),             _all_1q())

        # Two-qubit
        tgt.add_instruction(CUGate(theta, phi, lam, gam), _all_2q())
        tgt.add_instruction(CXGate(),              _all_2q())
        tgt.add_instruction(CYGate(),              _all_2q())
        tgt.add_instruction(CZGate(),              _all_2q())
        tgt.add_instruction(CHGate(),              _all_2q())
        tgt.add_instruction(CPhaseGate(theta),     _all_2q())
        tgt.add_instruction(CSXGate(),             _all_2q())
        tgt.add_instruction(SwapGate(),            _all_2q())
        tgt.add_instruction(iSwapGate(),           _all_2q())

        # Three-qubit
        tgt.add_instruction(CCXGate(),             _all_3q())
        tgt.add_instruction(CCZGate(),             _all_3q())
        tgt.add_instruction(CSwapGate(),           _all_3q())

        # Measure / reset
        tgt.add_instruction(Measure(), {(q,): InstructionProperties() for q in range(n)})
        tgt.add_instruction(Reset(),   {(q,): InstructionProperties() for q in range(n)})

        self._target = tgt
        return self._target

    def run(self, run_input, **options):
        """Run a QuantumCircuit (or list) on this backend."""
        opts = dict(self._options._fields)
        opts.update(options)

        qrack_options = {
            'is_schmidt_decompose_multi': opts.get('is_schmidt_decompose_multi'),
            'is_stabilizer_hybrid': (
                opts.get('is_stabilizer_hybrid') or
                opts.get('is_approx_near_clifford')
            ),
            'is_binary_decision_tree':    opts.get('is_binary_decision_tree'),
            'is_gpu':                     opts.get('is_gpu'),
            'is_near_clifford_tableau_writer': opts.get('is_near_clifford_cpu'),
            'is_host_pointer':            opts.get('is_host_pointer'),
            'noise':                      opts.get('noise'),
        }

        self._shots       = opts.get('shots', 1024)
        self._sdrp        = opts.get('sdrp', 0)
        self._is_approx_rz = opts.get('is_approx_near_clifford', False)
        self._is_t         = opts.get('is_t_injected', False)
        self._is_reactive  = opts.get('is_reactively_separated', False)

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

        self._number_of_qubits = len(experiment.qubits)
        self._number_of_clbits = len(experiment.clbits)

        # --- Build instruction list using Qiskit v2 CircuitInstruction API ---
        instructions = []
        for datum in experiment.data:
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

        # --- Preamble ---
        preamble_memory   = 0
        preamble_register = 0
        preamble_sim      = None

        def _make_sim():
            sim = QrackSimulator(qubit_count=self._number_of_qubits, **options)
            if self._sdrp > 0:
                sim.set_sdrp(self._sdrp)
            sim.set_use_exact_near_clifford(not self._is_approx_rz)
            sim.set_t_injection(self._is_t)
            sim.set_reactive_separate(self._is_reactive)
            return sim

        if boundary_start > 1:
            self._sim = _make_sim()
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
                self._sim = _make_sim()
                self._classical_memory   = 0
                self._classical_register = 0
            else:
                self._sim = QrackSimulator(cloneSid=preamble_sim.sid)
                if self._sdrp > 0:
                    self._sim.set_sdrp(self._sdrp)
                self._sim.set_use_exact_near_clifford(not self._is_approx_rz)
                self._sim.set_t_injection(self._is_t)
                self._sim.set_reactive_separate(self._is_reactive)
                self._classical_memory   = preamble_memory
                self._classical_register = preamble_register

            for op in instructions[boundary_start:]:
                self._apply_op(op)

            if not self._sample_measure and len(self._sample_qubits) > 0:
                self._data += [
                    bin(self._classical_memory)[2:].zfill(self._number_of_qubits)
                ]
                self._sample_qubits   = []
                self._sample_clbits   = []
                self._sample_cregbits = []

        if self._sample_measure and len(self._sample_qubits) > 0:
            self._data = self._add_sample_measure(
                self._sample_qubits, self._sample_clbits, self._shots
            )

        counts = dict(Counter(self._data))
        hex_counts = {'0x%x' % int(k, 2): v for k, v in counts.items()}

        return ExperimentResult(
            shots=self._shots,
            success=True,
            data=ExperimentResultData(counts=hex_counts, memory=self._data),
            status='DONE',
            header={
                'name': experiment.name,
                'n_qubits': self._number_of_qubits,
                'creg_sizes': [[r.name, r.size] for r in experiment.cregs],
                'memory_slots': self._number_of_clbits,
            },
        )

    def _apply_op(self, operation):
        name   = operation['name']
        qubits = operation['qubits']
        params = operation['params']

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

        if name in ('u1', 'p'):
            self._sim.u(qubits[0], 0, 0, float(params[0]))
        elif name == 'u2':
            self._sim.u(qubits[0], math.pi / 2, float(params[0]), float(params[1]))
        elif name in ('u3', 'u'):
            self._sim.u(qubits[0], float(params[0]), float(params[1]), float(params[2]))
        elif name == 'r':
            p1 = float(params[1])
            self._sim.u(qubits[0], float(params[0]), p1 - math.pi/2, -p1 + math.pi/2)
        elif name == 'unitary' and len(qubits) == 1:
            self._sim.mtrx(params[0].flatten(), qubits[0])
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
        elif name == 'sx':
            self._sim.mtrx([(1+1j)/2, (1-1j)/2, (1-1j)/2, (1+1j)/2], qubits[0])
        elif name == 'sxdg':
            self._sim.mtrx([(1-1j)/2, (1+1j)/2, (1+1j)/2, (1-1j)/2], qubits[0])
        elif name == 's':
            self._sim.s(qubits[0])
        elif name == 'sdg':
            self._sim.adjs(qubits[0])
        elif name == 't':
            self._sim.t(qubits[0])
        elif name == 'tdg':
            self._sim.adjt(qubits[0])
        elif name == 'cu1':
            self._sim.mcu(qubits[0:1], qubits[1], 0, 0, float(params[0]))
        elif name == 'cu3':
            self._sim.mcu(qubits[0:1], qubits[1],
                          float(params[0]), float(params[1]), float(params[2]))
        elif name == 'cu':
            self._sim.mcu(qubits[0:1], qubits[1],
                          float(params[0]), float(params[1]),
                          float(params[2]), float(params[3]))
        elif name == 'cx':
            self._sim.mcx(qubits[0:1], qubits[1])
        elif name == 'cy':
            self._sim.mcy(qubits[0:1], qubits[1])
        elif name == 'cz':
            self._sim.mcz(qubits[0:1], qubits[1])
        elif name == 'ch':
            self._sim.mch(qubits[0:1], qubits[1])
        elif name == 'cp':
            self._sim.mcmtrx(qubits[0:1],
                              [1, 0, 0, np.exp(1j * float(params[0]))],
                              qubits[1])
        elif name == 'csx':
            self._sim.mcmtrx(qubits[0:1],
                              [(1+1j)/2, (1-1j)/2, (1-1j)/2, (1+1j)/2],
                              qubits[1])
        elif name == 'dcx':
            self._sim.mcx(qubits[0:1], qubits[1])
            self._sim.mcx(qubits[1:2], qubits[0])
        elif name == 'ccx':
            self._sim.mcx(qubits[0:2], qubits[2])
        elif name == 'ccz':
            self._sim.mcz(qubits[0:2], qubits[2])
        elif name == 'swap':
            self._sim.swap(qubits[0], qubits[1])
        elif name == 'iswap':
            self._sim.iswap(qubits[0], qubits[1])
        elif name in ('iswap_dg', 'adjiswap'):
            self._sim.adjiswap(qubits[0], qubits[1])
        elif name == 'cswap':
            self._sim.cswap(qubits[0:1], qubits[1], qubits[2])
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
            sample = self._sim.m_all()
            result = 0
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
            data.append(bin(self._classical_memory)[2:].zfill(self._number_of_qubits))

        return data
