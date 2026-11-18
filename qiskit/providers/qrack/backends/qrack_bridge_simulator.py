# (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
#
# Qiskit BackendV2 for QrackBridge (bridged patching on native Qrack patches).
#
# By (Anthropic) Claude and Dan Strano
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

"""QrackBridgeSimulator: a Qiskit v2 backend over pyqrack.QrackBridge.

Topology (options n_qubits, patches, wrap) defines the Target: every qubit pair carries every native
2-qubit coupler, with InstructionProperties.error = bridge_error * (cuts crossed), so Qiskit's
error-aware layout (VF2Layout / VF2PostLayout) prefers in-patch and nearest-patch placement. For native
in-patch multi-control gates use backend.transpile() / generate_qrack_bridge_pass_manager() instead
of qiskit.transpile().

Measurement is terminal per qubit (QrackBridge samples the final state exactly; it has no collapse).
"""

import time
import uuid
from datetime import datetime

from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    CHGate, CPhaseGate, CRXGate, CRYGate, CRZGate, CSXGate, CUGate, CXGate, CYGate, CZGate,
    HGate, IGate, Measure, PhaseGate, Reset, RXGate, RYGate, RZGate, SdgGate, SGate, SwapGate,
    SXdgGate, SXGate, TdgGate, TGate, U1Gate, U2Gate, U3Gate, UGate, XGate, YGate, ZGate, iSwapGate,
)
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.providers.backend import BackendV2
from qiskit.providers.options import Options
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.transpiler import InstructionProperties, Target

from pyqrack import QrackBridge

from ..qrackerror import QrackError
from ..qrackjob import QrackJob
from ..version import __version__
from .qrack_bridge_gates import FSimGate, ISwapDgGate
from .qrack_bridge_transpiler import (
    bridge_partition,
    generate_qrack_bridge_pass_manager,
    patch_distance_matrix,
)

_TOPOLOGY_OPTIONS = ("n_qubits", "patches", "wrap", "bridge_error")
_ENGINE_OPTIONS = (
    "is_tensor_network", "is_schmidt_decompose", "is_schmidt_decompose_multi", "is_stabilizer_hybrid",
    "is_binary_decision_tree", "is_paged", "is_cpu_gpu_hybrid", "is_gpu",
)


class QrackBridgeSimulator(BackendV2):
    """Qiskit v2 backend for pyqrack.QrackBridge (see module docstring)."""

    DEFAULT_OPTIONS = {
        # Qiskit only allows many qubits for certain methods; this is not Aer's MPS.
        "method": "matrix_product_state",
        "shots": 1024,
        "memory": False,
        # topology (changing these rebuilds the Target)
        "n_qubits": 32,
        "patches": 2,
        "wrap": False,
        "bridge_error": 1e-3,  # Target error per cut a bridged coupler crosses (layout heuristic only)
        # QrackBridge
        "max_bonds": 6,
        "truncation": "greedy",
        "lazy_bonds": 0,
        "max_patch_qubits": None,
        "seed": None,  # experiment k of a job uses seed + k
        "is_tensor_network": True,
        "is_schmidt_decompose": True,
        "is_schmidt_decompose_multi": False,
        "is_stabilizer_hybrid": False,
        "is_binary_decision_tree": False,
        "is_paged": True,
        "is_cpu_gpu_hybrid": True,
        "is_gpu": True,
    }

    max_circuits = None

    def __init__(self, provider=None, **fields):
        for field in fields:
            if field not in self.DEFAULT_OPTIONS:
                raise AttributeError("Options field %s is not valid for this backend" % field)
        super().__init__(
            provider=provider,
            name="qrack_bridge_simulator",
            description="Qrack bridged-patch (block MPS of state-vector patches) simulator",
            backend_version=__version__,
        )
        self._options = self._default_options()
        if fields:
            self._options.update_options(**fields)
        self._target = None
        self._check_topology()

    @classmethod
    def _default_options(cls):
        opts = Options()
        opts.update_options(**cls.DEFAULT_OPTIONS)
        return opts

    def set_options(self, **fields):
        super().set_options(**fields)
        if any(f in _TOPOLOGY_OPTIONS for f in fields):
            self._target = None
            self._check_topology()

    def _check_topology(self):
        o = self._options
        if not 1 <= o.patches <= o.n_qubits:
            raise QrackError("QrackBridgeSimulator needs 1 <= patches <= n_qubits.")

    # --- topology -----------------------------------------------------------------

    @property
    def partition(self):
        """Physical qubits of each patch, as QrackBridge allocates them."""
        return bridge_partition(self._options.n_qubits, self._options.patches)

    def patch_of(self, q):
        for p, qs in enumerate(self.partition):
            if q in qs:
                return p
        raise IndexError(q)

    @property
    def target(self):
        if self._target is not None:
            return self._target
        o = self._options
        n = o.n_qubits
        part = self.partition
        patch = [p for p, qs in enumerate(part) for _ in qs]
        D = patch_distance_matrix(o.patches, o.wrap)
        tgt = Target(num_qubits=n, description=self.description)

        th, ph, lm, gm = Parameter("theta"), Parameter("phi"), Parameter("lam"), Parameter("gamma")

        def one():
            return {(q,): InstructionProperties(error=0.0) for q in range(n)}

        def two():
            return {
                (a, b): InstructionProperties(error=float(o.bridge_error * D[patch[a], patch[b]]))
                for a in range(n) for b in range(n) if a != b
            }

        for gate in (
            IGate(), UGate(th, ph, lm), U3Gate(th, ph, lm), U2Gate(ph, lm), U1Gate(lm), PhaseGate(th),
            RXGate(th), RYGate(th), RZGate(th), HGate(), XGate(), YGate(), ZGate(),
            SGate(), SdgGate(), SXGate(), SXdgGate(), TGate(), TdgGate(),
        ):
            tgt.add_instruction(gate, one())

        # Controlled single-qubit gates: QrackBridge.mcmtrx (one control may cross patches).
        for gate in (
            CXGate(), CYGate(), CZGate(), CHGate(), CPhaseGate(th), CRXGate(th), CRYGate(th),
            CRZGate(th), CUGate(th, ph, lm, gm), CSXGate(),
        ):
            tgt.add_instruction(gate, two())
        # Open (anti-)controls: QrackBridge.macmtrx.
        for gate in (CXGate(ctrl_state=0), CYGate(ctrl_state=0), CZGate(ctrl_state=0)):
            tgt.add_instruction(gate, two(), name=gate.name)
        # Ship-and-return couplers.
        tgt.add_instruction(SwapGate(), two())
        tgt.add_instruction(iSwapGate(), two())
        tgt.add_instruction(ISwapDgGate(), two())
        tgt.add_instruction(FSimGate(th, ph), two())

        tgt.add_instruction(Measure(), one())
        tgt.add_instruction(Reset(), one())  # only before a qubit's first gate (a no-op on |0>)

        self._target = tgt
        return tgt

    # --- transpilation --------------------------------------------------------------

    def get_pass_manager(self, optimization_level=1, **kwargs):
        """generate_qrack_bridge_pass_manager() for this backend's topology."""
        return generate_qrack_bridge_pass_manager(self, optimization_level=optimization_level, **kwargs)

    def transpile(self, circuits, optimization_level=1, **kwargs):
        """Patch-aware transpile keeping native in-patch multi-control gates (see module docstring)."""
        return self.get_pass_manager(optimization_level, **kwargs).run(circuits)

    # --- execution --------------------------------------------------------------------

    def run(self, run_input, **options):
        """Run a QuantumCircuit (or list of them); each starts from |0...0> on a fresh bridge."""
        for field in options:
            if field not in self.DEFAULT_OPTIONS:
                raise AttributeError("Options field %s is not valid for this backend" % field)
        opts = dict(self._options.items())
        opts.update(options)
        if any(f in options for f in ("n_qubits", "patches", "wrap")):
            raise QrackError("Change topology with set_options() so the Target matches; not in run().")

        if isinstance(run_input, QuantumCircuit):
            circuits = [run_input]
        elif isinstance(run_input, (list, tuple)):
            circuits = list(run_input)
        else:
            raise QrackError("Unrecognized run_input type: %s" % type(run_input))

        job_id = str(uuid.uuid4())
        result = self._run_job(job_id, circuits, opts)
        return QrackJob(self, job_id, result, run_input)

    def _make_bridge(self, opts, k):
        seed = opts.get("seed")
        return QrackBridge(
            opts["n_qubits"],
            max_bonds=opts["max_bonds"],
            seed=None if seed is None else int(seed) + k,
            truncation=opts["truncation"],
            patches=opts["patches"],
            wrap=opts["wrap"],
            lazy_bonds=opts["lazy_bonds"],
            max_patch_qubits=opts["max_patch_qubits"],
            **{f: opts[f] for f in _ENGINE_OPTIONS},
        )

    def _run_job(self, job_id, circuits, opts):
        results = []
        for k, circ in enumerate(circuits):
            if not isinstance(circ, QuantumCircuit):
                raise QrackError("run_input must be QuantumCircuit(s).")
            results.append(self._run_experiment(circ, opts, k))
        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            job_id=job_id,
            success=True,
            results=results,
            date=datetime.now().isoformat(),
            status="COMPLETED",
            header={},
        )

    def _run_experiment(self, circ, opts, k):
        start = time.time()
        if circ.num_qubits > opts["n_qubits"]:
            raise QrackError(
                "Circuit %s has %d qubits; this backend has n_qubits=%d."
                % (circ.name, circ.num_qubits, opts["n_qubits"])
            )
        shots = int(opts["shots"])
        bridge = self._make_bridge(opts, k)
        try:
            samples = bridge.run_qiskit_circuit(circ, shots=shots)
        except RuntimeError as e:
            raise QrackError(str(e)) from e

        counts = {}
        hex_memory = []
        for s in samples:
            h = hex(int(s, 2))
            counts[h] = counts.get(h, 0) + 1
            hex_memory.append(h)
        data = {"counts": counts}
        if opts.get("memory"):
            data["memory"] = hex_memory

        return ExperimentResult(
            shots=shots,
            success=True,
            data=ExperimentResultData(**data),
            status="DONE",
            seed=opts.get("seed"),
            header={
                "name": circ.name,
                "n_qubits": circ.num_qubits,
                "creg_sizes": [[r.name, r.size] for r in circ.cregs],
                "clbit_labels": [[r.name, i] for r in circ.cregs for i in range(r.size)],
                "memory_slots": circ.num_clbits,
                "metadata": circ.metadata,
            },
            time_taken=time.time() - start,
            metadata={
                "fidelity_estimate": bridge.fidelity_estimate,
                "truncations": bridge.truncations,
                "flushes": bridge.flushes,
                "peak_bonds": bridge.peak_bonds,
                "peak_patch_qubits": bridge.peak_patch_qubits,
                "bond": bridge.bond,
            },
        )
