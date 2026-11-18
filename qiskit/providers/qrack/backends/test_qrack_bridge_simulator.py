# Tests for QrackBridge.run_qiskit_circuit(), QrackBridgeSimulator, and its transpiler passes.
import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import CCZGate, CXGate, MCXGate, iSwapGate
from qiskit.quantum_info import Operator, Statevector, random_unitary
from pyqrack import QrackBridge
from qiskit.providers.qrack import FSimGate, ISwapDgGate, QrackBridgeSimulator
from qiskit.providers.qrack.backends.qrack_bridge_transpiler import patch_distance_matrix


def rand_circ(n, depth, seed, mc=True):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(depth):
        q = [int(x) for x in rng.choice(n, 4, replace=False)]
        a = rng.uniform(-3, 3, 4)
        ops = [lambda: qc.u(*a[:3], q[0]), lambda: qc.sx(q[0]), lambda: qc.cx(q[0], q[1]),
               lambda: qc.append(CXGate(ctrl_state=0), q[:2]), lambda: qc.cu(*a, q[0], q[1]),
               lambda: qc.rzz(a[0], q[0], q[1]), lambda: qc.iswap(q[0], q[1]),
               lambda: qc.append(iSwapGate().inverse(), q[:2]), lambda: qc.append(FSimGate(a[0], a[1]), q[:2]),
               lambda: qc.append(ISwapDgGate(), q[:2]), lambda: qc.crx(a[0], q[0], q[1]),
               lambda: qc.unitary(random_unitary(4, seed=int(rng.integers(1e6))), q[:2])]
        if mc:
            ops += [lambda: qc.ccx(*q[:3]), lambda: qc.append(MCXGate(3, ctrl_state=5), q),
                    lambda: qc.mcp(a[0], q[:3], q[3]), lambda: qc.append(CCZGate(ctrl_state=2), q[:3]),
                    lambda: qc.cswap(*q[:3])]
        ops[rng.integers(len(ops))]()
    return qc


def test_gate_primitives():
    g = FSimGate(0.3, 0.7)
    assert np.allclose(Operator(g.definition).data, np.array(g))
    assert Operator(FSimGate(-np.pi / 2, 0)).equiv(iSwapGate())
    d = ISwapDgGate()
    assert np.allclose(Operator(d.definition).data, np.array(d))
    assert np.allclose(np.array(d), Operator(iSwapGate()).data.conj().T)


@pytest.mark.parametrize("patches,wrap", [(1, False), (2, False), (3, False)])
def test_run_qiskit_circuit_statevector(patches, wrap):
    for seed in range(3):
        qc = rand_circ(8, 40, seed)
        b = QrackBridge(8, max_bonds=8, patches=patches, wrap=wrap, is_gpu=False)
        b.run_qiskit_circuit(qc)
        assert abs(np.vdot(Statevector(qc).data, b.out_ket())) ** 2 > 1 - 1e-5


def test_terminal_measurement_and_errors():
    qc = QuantumCircuit(4, 3)
    qc.h(0); qc.cx(0, 1); qc.cx(1, 3); qc.measure(0, 0); qc.x(2); qc.measure([3, 2], [1, 2])
    assert set(QrackBridge(4, is_gpu=False).run_qiskit_circuit(qc, shots=200)) == {"100", "111"}
    bad = QuantumCircuit(1, 1); bad.h(0); bad.measure(0, 0); bad.h(0)
    with pytest.raises(RuntimeError):
        QrackBridge(1, is_gpu=False).run_qiskit_circuit(bad)


def _pad(qc, n):
    p = QuantumCircuit(n); p.compose(qc, range(qc.num_qubits), inplace=True); return p


def test_transpile_equivalence_and_target():
    be = QrackBridgeSimulator(n_qubits=9, patches=3, is_gpu=False)
    qc = rand_circ(7, 60, 11)
    U = Operator(_pad(qc, 9))
    names = set(be.target.operation_names)
    for lvl in range(4):
        t = transpile(qc, be, optimization_level=lvl, seed_transpiler=3)
        assert set(t.count_ops()) <= names and Operator.from_circuit(t).equiv(U)
    patch = {q: p for p, qs in enumerate(be.partition) for q in qs}
    for lvl in range(3):
        t = be.transpile(qc, optimization_level=lvl)
        assert Operator.from_circuit(t).equiv(U)
        for i in t.data:
            qs = [t.find_bit(q).index for q in i.qubits]
            if len(qs) >= 3:
                assert len({patch[x] for x in qs}) == 1


def test_patch_layout_regroups_clusters():
    rng = np.random.default_rng(3)
    be = QrackBridgeSimulator(n_qubits=12, patches=3, is_gpu=False)
    clusters = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
    qc = QuantumCircuit(12)
    for _ in range(120):
        a, b, c = [int(x) for x in rng.choice(clusters[rng.integers(3)], 3, replace=False)]
        [lambda: qc.cx(a, b), lambda: qc.ry(0.3, a), lambda: qc.ccx(a, b, c)][rng.integers(3)]()
    t = be.transpile(qc)
    patch = {q: p for p, qs in enumerate(be.partition) for q in qs}
    D = patch_distance_matrix(3)
    assert sum(D[patch[t.find_bit(i.qubits[0]).index], patch[t.find_bit(i.qubits[1]).index]]
               for i in t.data if len(i.qubits) == 2) == 0
    assert t.count_ops().get("ccx", 0) == qc.count_ops()["ccx"]


def test_backend_counts():
    be = QrackBridgeSimulator(n_qubits=9, patches=3, is_gpu=False, max_bonds=10, seed=5)
    qc = rand_circ(7, 60, 11)
    t = be.transpile(qc, optimization_level=2)
    m = t.copy(); m.add_register(ClassicalRegister(7))
    phys = t.layout.final_index_layout()
    m.measure([phys[i] for i in range(7)], range(7))
    counts = be.run(m, shots=40000).result().get_counts()
    probs = Statevector(qc).probabilities()
    keys = set(counts) | {format(i, "07b") for i in range(128)}
    assert 0.5 * sum(abs(counts.get(k, 0) / 40000 - probs[int(k, 2)]) for k in keys) < 0.04
    r = be.run(transpile(QuantumCircuit(1, 1).compose(QuantumCircuit(1)), be), shots=3, memory=True).result()
    assert r.get_memory() == ["0", "0", "0"]
