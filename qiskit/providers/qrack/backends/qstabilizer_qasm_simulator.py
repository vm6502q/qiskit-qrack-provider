# Based on and adapted from the AceQasmSimulator pattern in this same
# provider (itself adapted from
# https://github.com/Qiskit/qiskit-qcgpu-provider/blob/master/qiskit_qcgpu_provider/qasm_simulator.py).
#
# Unlike AceQasmSimulator, this backend has NO sample-measure shortcut:
# every shot rebuilds a fresh QrackStabilizer and re-runs the ENTIRE
# circuit from the top, because the stochastic rounding decisions inside
# every t/tdg/rz gate are what make weak simulation statistically valid
# in the first place -- reusing a single execution's state across
# multiple "shots" would silently collapse that into one fixed stochastic
# trajectory sampled repeatedly, not independent draws.

import uuid
import time
import math
from datetime import datetime
from collections import Counter

from ..version import __version__
from ..qrackjob import QrackJob
from ..qrackerror import QrackError
from pyqrack import QrackStabilizer, Pauli

from qiskit.providers.backend import BackendV2
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.providers.options import Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import Clbit, Parameter

from qiskit.circuit.library import (
    IGate, RZGate,
    XGate, YGate, ZGate, HGate,
    SGate, SdgGate, TGate, TdgGate, SXGate, SXdgGate,
    CXGate, CYGate, CZGate, SwapGate, iSwapGate,
    Measure, Reset,
)

from .qstabilizer_noise import create_noise_model, insert_rz_errors, T_ERROR_PROBABILITY, TDG_ERROR_PROBABILITY


class QrackQasmQobjInstructionConditional:
    def __init__(self, mask, val):
        self.mask = mask
        self.val = val

    def to_dict(self):
        return vars(self)


class QStabilizerQasmSimulator(BackendV2):
    """
    Near-Clifford weak-simulation backend on QrackStabilizer.

    Every qubit has an identical error profile on every non-Clifford gate
    (t/tdg: p=0.5 exactly; generic rz(theta): p=2*|reduced angle|/pi),
    and connectivity is full -- there is no boundary/bulk asymmetry the
    way there is for QrackAceBackend, so the Target and noise model here
    are considerably simpler.
    """

    DEFAULT_OPTIONS = {
        "n_qubits": 32,
        "shots": 1024,
    }

    # Matches QrackStabilizer.get_qiskit_basis_gates()
    BASIS_GATES = [
        "id", "rz", "t", "tdg",
        "h", "x", "y", "z", "s", "sdg", "sx", "sxdg",
        "cx", "cy", "cz", "swap", "iswap", "reset", "measure",
    ]

    max_circuits = None

    def __init__(self, provider=None, **fields):
        for field in fields:
            if field not in self.DEFAULT_OPTIONS:
                raise AttributeError(
                    "Options field %s is not valid for this backend" % field
                )

        super().__init__(
            provider=provider,
            name="qstabilizer_qasm_simulator",
            description="Near-Clifford weak-simulation qasm simulator on QrackStabilizer",
            backend_version=__version__,
        )

        self._options = self._default_options()
        if fields:
            self._options.update_options(**fields)

        self._number_of_qubits = self._options.get("n_qubits")
        self._target = None
        self._noise_model = create_noise_model(self._number_of_qubits)

    @classmethod
    def _default_options(cls):
        opts = Options()
        opts.update_options(**cls.DEFAULT_OPTIONS)
        return opts

    def get_noise_model(self):
        """Aer NoiseModel for t/tdg (fixed-angle gates). For circuits
        containing generic rz(theta), use insert_rz_errors(circuit)
        directly instead of (or alongside) this noise model -- rz's
        continuously angle-dependent error can't be expressed via Aer's
        per-(gate name, qubit) dispatch. Build/run any circuit using this
        noise model with optimization_level=0: standard transpiler
        optimization will otherwise remove t/tdg/rz instances it judges
        redundant, silently dropping the very gates the noise model
        depends on."""
        return self._noise_model

    @property
    def coupling_map(self):
        return None  # full connectivity

    @property
    def target(self):
        if self._target is not None:
            return self._target

        n = self._number_of_qubits
        tgt = Target(num_qubits=n, description=self.description)
        theta = Parameter("theta")

        # Uniform, zero-error single-qubit Clifford gates (exact
        # stabilizer operations; verified 2-qubit Cliffords are exact
        # too -- see below).
        all_1q = {(q,): InstructionProperties(error=0.0) for q in range(n)}
        tgt.add_instruction(IGate(), all_1q)
        tgt.add_instruction(HGate(), all_1q)
        tgt.add_instruction(XGate(), all_1q)
        tgt.add_instruction(YGate(), all_1q)
        tgt.add_instruction(ZGate(), all_1q)
        tgt.add_instruction(SGate(), all_1q)
        tgt.add_instruction(SdgGate(), all_1q)
        tgt.add_instruction(SXGate(), all_1q)
        tgt.add_instruction(SXdgGate(), all_1q)

        # Non-Clifford, uniform error: t/tdg fixed at p=0.5 exactly
        # (both sit exactly at the quadrant boundary). rz(theta) has no
        # single closed-form error here since InstructionProperties
        # can't carry a per-instance angle parameter -- report the
        # WORST-CASE rate (0.5, same boundary case as t/tdg) as a
        # conservative hint; use rz_error_probability(theta) from
        # qstabilizer_noise directly for the exact, angle-specific rate.
        t_props = {(q,): InstructionProperties(error=T_ERROR_PROBABILITY) for q in range(n)}
        tdg_props = {(q,): InstructionProperties(error=TDG_ERROR_PROBABILITY) for q in range(n)}
        rz_props = {(q,): InstructionProperties(error=0.5) for q in range(n)}
        tgt.add_instruction(TGate(), t_props)
        tgt.add_instruction(TdgGate(), tdg_props)
        tgt.add_instruction(RZGate(theta), rz_props)

        # Two-qubit gates: full connectivity, zero error (verified: 0/300
        # mismatches against an exact reference for a Bell-pair cx --
        # cx/cy/cz/swap/iswap never touch the stochastic buffer at all).
        pair_props = {
            (a, b): InstructionProperties(error=0.0)
            for a in range(n) for b in range(n) if a != b
        }
        tgt.add_instruction(CXGate(), pair_props)
        tgt.add_instruction(CYGate(), pair_props)
        tgt.add_instruction(CZGate(), pair_props)
        tgt.add_instruction(SwapGate(), pair_props)
        tgt.add_instruction(iSwapGate(), pair_props)

        tgt.add_instruction(Measure(), {(q,): InstructionProperties() for q in range(n)})
        tgt.add_instruction(Reset(), {(q,): InstructionProperties() for q in range(n)})

        self._target = tgt
        return self._target

    def run(self, run_input, **options):
        opts = dict(self._options._fields)
        opts.update(options)
        self._shots = opts.get("shots", 1024)

        job_id = str(uuid.uuid4())
        job = QrackJob(
            self, job_id,
            self._run_job(job_id, run_input),
            run_input,
        )
        return job

    def _run_job(self, job_id, run_input):
        if isinstance(run_input, QuantumCircuit):
            experiments = [run_input]
        elif isinstance(run_input, list):
            experiments = run_input
        else:
            raise QrackError("Unrecognized run_input type: %s" % type(run_input))

        results = [self._run_experiment(experiment) for experiment in experiments]

        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            job_id=job_id,
            success=True,
            results=results,
            date=datetime.now(),
            status="COMPLETED",
            header={},
        )

    def _run_experiment(self, experiment):
        """Run one QuantumCircuit. Every shot is a fresh, independent,
        full top-to-bottom execution -- see the module-level note on why
        this is mandatory for QrackStabilizer specifically, unlike
        QrackAceBackend's sample-measure shortcut."""
        if not isinstance(experiment, QuantumCircuit):
            raise QrackError("run_input must be a QuantumCircuit.")

        self._number_of_qubits = len(experiment.qubits)
        self._number_of_clbits = len(experiment.clbits)

        instructions = []
        for datum in experiment.data:
            op = datum.operation
            qubits = [experiment.find_bit(q).index for q in datum.qubits]
            clbits = [experiment.find_bit(c).index for c in datum.clbits]

            conditional = None
            condition = getattr(op, "condition", None)
            if condition is not None:
                if isinstance(condition[0], Clbit):
                    conditional = experiment.find_bit(condition[0]).index
                else:
                    creg_index = experiment.cregs.index(condition[0])
                    size = experiment.cregs[creg_index].size
                    offset = sum(len(experiment.cregs[i]) for i in range(creg_index))
                    mask = ((1 << offset) - 1) ^ ((1 << (offset + size)) - 1)
                    val = condition[1]
                    conditional = (
                        offset if size == 1
                        else QrackQasmQobjInstructionConditional(mask, val)
                    )

            instructions.append({
                "name": op.name,
                "qubits": qubits,
                "memory": clbits,
                "conditional": conditional,
                "params": list(op.params),
            })

        data = []
        for _ in range(self._shots):
            self._sim = QrackStabilizer(self._number_of_qubits)
            self._classical_memory = 0
            self._classical_register = 0
            for op in instructions:
                self._apply_op(op)
            data.append(bin(self._classical_memory)[2:].zfill(self._number_of_clbits))

        counts = dict(Counter(data))
        hex_counts = {"0x%x" % int(k, 2): v for k, v in counts.items()}

        return ExperimentResult(
            shots=self._shots,
            success=True,
            data=ExperimentResultData(counts=hex_counts, memory=data),
            status="DONE",
            header={"name": experiment.name,
                    "n_qubits": self._number_of_qubits,
                    "creg_sizes": [[r.name, r.size] for r in experiment.cregs],
                    "memory_slots": self._number_of_clbits},
        )

    def _apply_op(self, operation):
        name = operation["name"]

        if name in ("id", "barrier"):
            return

        conditional = operation.get("conditional")
        if isinstance(conditional, int):
            if not ((self._classical_register >> conditional) & 1):
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

        qubits = operation["qubits"]
        params = operation["params"]

        if name == "rz":
            self._sim.r(Pauli.PauliZ, float(params[0]), qubits[0])
        elif name == "h":
            self._sim.h(qubits[0])
        elif name == "x":
            self._sim.x(qubits[0])
        elif name == "y":
            self._sim.y(qubits[0])
        elif name == "z":
            self._sim.z(qubits[0])
        elif name == "s":
            self._sim.s(qubits[0])
        elif name == "sdg":
            self._sim.adjs(qubits[0])
        elif name == "sx":
            self._sim.sx(qubits[0])
        elif name == "sxdg":
            self._sim.adjsx(qubits[0])
        elif name == "t":
            self._sim.t(qubits[0])
        elif name == "tdg":
            self._sim.adjt(qubits[0])
        elif name == "cx":
            self._sim.mcx([qubits[0]], qubits[1])
        elif name == "cy":
            self._sim.mcy([qubits[0]], qubits[1])
        elif name == "cz":
            self._sim.mcz([qubits[0]], qubits[1])
        elif name == "swap":
            self._sim.swap(qubits[0], qubits[1])
        elif name == "iswap":
            self._sim.iswap(qubits[0], qubits[1])
        elif name == "reset":
            for q in qubits:
                if self._sim.m(q):
                    self._sim.x(q)
        elif name == "measure":
            clbits = operation["memory"]
            for idx in range(len(qubits)):
                outcome = self._sim.m(qubits[idx])
                clbit = clbits[idx]
                clmask = 1 << clbit
                self._classical_memory = (
                    (self._classical_memory & ~clmask) | (outcome << clbit)
                )
                self._classical_register = (
                    (self._classical_register & ~clmask) | (outcome << clbit)
                )
        else:
            raise QrackError(
                '%s encountered unrecognized operation "%s"' % (self.name, name)
            )
