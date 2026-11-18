# (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
#
# Transpiler passes and a pass manager for QrackBridge.
#
# By (Anthropic) Claude and Dan Strano
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

"""Patch-aware transpilation for QrackBridge.

QrackBridge couples any two qubits exactly, but a gate between patches p and q leaves a bond on every
cut between them (|p - q| cuts on a chain, the shorter way round on a ring), and multi-control gates
run natively only when all of their qubits share a patch. Qiskit's preset pass managers cannot keep
3+-qubit gates on restricted qubit sets (levels 0-1 pass them through unchecked; levels 2-3 refuse the
Target), so QrackBridgeSimulator.target lists 1- and 2-qubit couplers only. The pass manager here
keeps in-patch multi-control gates instead:

    1. QrackBridgeUnroll: expand definitions down to what QrackBridge runs natively.
    2. QrackBridgePatchLayout: place virtual qubits in patches to minimise cut-weighted interactions,
       pulling the operands of each multi-control gate into one patch.
    3. QrackBridgeSplitCrossPatchControls: expand only the multi-control gates still spanning patches.
"""

import numpy as np

from qiskit.circuit import ControlledGate, Gate
from qiskit.circuit.library import (
    CXGate, CYGate, CZGate, HGate, SwapGate, XGate, YGate, ZGate, iSwapGate,
)
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap, Layout, PassManager, TranspilerError
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from qiskit.transpiler.passes import (
    ApplyLayout,
    CommutativeCancellation,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    InverseCancellation,
    Optimize1qGatesDecomposition,
)

from .qrack_bridge_gates import ISwapDgGate

_PASSIVE = frozenset(("measure", "reset", "barrier", "delay", "id"))
_NATIVE_2Q = frozenset(("swap", "iswap", "iswap_dg", "fsim"))
_CONTROL_FLOW = frozenset(("if_else", "while_loop", "for_loop", "switch_case", "box"))


def bridge_partition(num_qubits, patches):
    """QrackBridge's patch blocks: contiguous, equal up to the remainder (the last ones get one more)."""
    base, rem = divmod(num_qubits, patches)
    sizes = [base + (1 if p >= patches - rem else 0) for p in range(patches)]
    start = [sum(sizes[:p]) for p in range(patches)]
    return [list(range(start[p], start[p] + sizes[p])) for p in range(patches)]


def bridge_is_ring(patches, wrap):
    return bool(wrap) and patches >= 3


def patch_distance_matrix(patches, wrap=False):
    """Cuts a bridged gate crosses between patches i and j (bonds it adds, one per cut)."""
    i, j = np.indices((patches, patches))
    d = np.abs(i - j)
    if bridge_is_ring(patches, wrap):
        d = np.minimum(d, patches - d)
    return d.astype(float)


def is_bridge_native(op):
    """True if QrackBridge.run_qiskit_circuit() runs op without expanding its definition
    (whether a multi-control gate also needs one patch is decided after layout)."""
    name = op.name
    if name in _PASSIVE or name in _NATIVE_2Q or name in _CONTROL_FLOW:
        return True
    if not isinstance(op, Gate):
        return False
    if op.num_qubits == 1:
        return True
    return isinstance(op, ControlledGate) and op.base_gate.num_qubits == 1


def _expand(dag, node):
    definition = node.op.definition
    if definition is None:
        raise TranspilerError('QrackBridge cannot run "%s" and it has no definition.' % node.op.name)
    dag.substitute_node_with_dag(node, circuit_to_dag(definition))


class QrackBridgeUnroll(TransformationPass):
    """Expand definitions until every operation is a QrackBridge native (see is_bridge_native)."""

    def run(self, dag):
        changed = True
        while changed:
            changed = False
            for node in dag.op_nodes():
                if not is_bridge_native(node.op):
                    _expand(dag, node)
                    changed = True
        return dag


class QrackBridgePatchLayout(AnalysisPass):
    """Choose a layout that keeps interacting qubits in the same or nearby patches.

    Minimises sum_gates sum_pairs w * d(patch_a, patch_b), with d the cut distance, w = 1 for 2-qubit
    gates and w = multi_control_weight per operand pair of a 3+-qubit gate (a split multi-control gate
    costs a full decomposition). Starts from the trivial layout and applies the best pairwise exchange
    (or move into a free slot) until none improves. Placement within a patch does not matter.
    """

    def __init__(self, num_qubits, patches=2, wrap=False, multi_control_weight=8.0, max_iterations=None):
        super().__init__()
        self.num_qubits = num_qubits
        self.patches = patches
        self.wrap = wrap
        self.multi_control_weight = multi_control_weight
        self.max_iterations = max_iterations

    def run(self, dag):
        n, w = self.num_qubits, dag.num_qubits()
        if w > n:
            raise TranspilerError("Circuit has %d qubits; the bridge has %d." % (w, n))
        index = {q: i for i, q in enumerate(dag.qubits)}
        W = np.zeros((n, n))
        for node in dag.op_nodes():
            qs = [index[q] for q in node.qargs]
            if len(qs) < 2 or node.op.name == "barrier":
                continue
            wt = 1.0 if len(qs) == 2 else self.multi_control_weight
            for a in range(len(qs)):
                for b in range(a + 1, len(qs)):
                    W[qs[a], qs[b]] += wt
                    W[qs[b], qs[a]] += wt

        part = bridge_partition(n, self.patches)
        slot_patch = np.array([p for p, qs in enumerate(part) for _ in qs])
        D = patch_distance_matrix(self.patches, self.wrap)
        p = slot_patch.copy()  # nodes 0..w-1 are virtual qubits; w..n-1 are free slots (W = 0)
        iters = self.max_iterations if self.max_iterations is not None else 8 * n
        for _ in range(iters):
            A = W @ D[p]  # A[i, X]: cost of node i's interactions if i sat in patch X
            own = A[np.arange(n), p]
            G = A[:, p]  # G[i, j] = A[i, p_j]
            delta = G - own[:, None] + G.T - own[None, :] + 2 * W * D[p][:, p]
            delta[p[:, None] == p[None, :]] = 0
            i, j = np.unravel_index(np.argmin(delta), delta.shape)
            if delta[i, j] > -1e-12:
                break
            p[i], p[j] = p[j], p[i]

        free = {pp: list(qs) for pp, qs in enumerate(part)}
        layout = Layout()
        for v in range(w):
            layout[dag.qubits[v]] = free[p[v]].pop(0)
        for qreg in dag.qregs.values():
            layout.add_register(qreg)
        self.property_set["layout"] = layout
        self.property_set["qrack_bridge_layout_cost"] = float(
            0.5 * np.sum(W * D[p][:, p])
        )


class QrackBridgeSplitCrossPatchControls(TransformationPass):
    """On a physical circuit, expand multi-control gates whose qubits do not share one patch."""

    def __init__(self, num_qubits, patches=2):
        super().__init__()
        self.patch_of = {
            q: pp for pp, qs in enumerate(bridge_partition(num_qubits, patches)) for q in qs
        }

    def run(self, dag):
        index = {q: i for i, q in enumerate(dag.qubits)}
        changed = True
        while changed:
            changed = False
            for node in dag.op_nodes():
                if len(node.qargs) < 3 or node.op.name in _PASSIVE or node.op.name in _CONTROL_FLOW:
                    continue
                if len({self.patch_of[index[q]] for q in node.qargs}) > 1 or not is_bridge_native(node.op):
                    _expand(dag, node)
                    changed = True
        return dag


def _optimization_passes(level):
    if level < 1:
        return []
    passes = [
        Optimize1qGatesDecomposition(basis=["u"]),
        InverseCancellation(
            [CXGate(), CYGate(), CZGate(), SwapGate(), HGate(), XGate(), YGate(), ZGate(),
             (iSwapGate(), ISwapDgGate())]
        ),
    ]
    if level >= 2:
        passes.append(CommutativeCancellation())
        passes.append(Optimize1qGatesDecomposition(basis=["u"]))
    return passes


def generate_qrack_bridge_pass_manager(
    backend=None,
    *,
    num_qubits=None,
    patches=None,
    wrap=None,
    optimization_level=1,
    multi_control_weight=8.0,
):
    """PassManager that maps circuits onto a QrackBridge, keeping native in-patch multi-control gates.

    Pass a QrackBridgeSimulator (topology read from its options) or num_qubits/patches/wrap directly.
    Its output runs on QrackBridgeSimulator.run() or QrackBridge.run_qiskit_circuit(); it may hold
    gates (ccx, mcx, crx, ...) that QrackBridgeSimulator.target omits, since Target cannot express
    "3+ qubits, one patch" in a form the preset passes accept.

    optimization_level: 0 (layout only), 1 (+ 1q fusion, inverse cancellation), 2+ (+ commutation).
    """
    if backend is not None:
        opts = backend.options
        num_qubits = opts.n_qubits if num_qubits is None else num_qubits
        patches = opts.patches if patches is None else patches
        wrap = opts.wrap if wrap is None else wrap
    if num_qubits is None:
        raise ValueError("Give a backend or num_qubits.")
    patches = 2 if patches is None else patches
    wrap = False if wrap is None else wrap

    pm = PassManager()
    pm.append(QrackBridgeUnroll())
    pm.append(_optimization_passes(optimization_level))
    pm.append(QrackBridgePatchLayout(num_qubits, patches, wrap, multi_control_weight))
    pm.append(FullAncillaAllocation(CouplingMap.from_full(num_qubits)))
    pm.append(EnlargeWithAncilla())
    pm.append(ApplyLayout())
    pm.append(QrackBridgeSplitCrossPatchControls(num_qubits, patches))
    pm.append(_optimization_passes(optimization_level))
    return pm
