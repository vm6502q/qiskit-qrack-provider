# Error model for QrackStabilizer's near-Clifford weak-simulation gate set.
#
# Mechanism (verified directly against the compiled library, Qrack commit
# 4024713136dbc2cfefec70a69c7abd6f964da4e7, with the FlipQuadrant fix):
# QStabilizer::RZ represents an arbitrary Z-rotation by first stripping
# whole pi/2 multiples EXACTLY as S/Sdg gates (zero cost), leaving a
# residual angle phi in [-pi/4, pi/4], then stochastically applying one
# more quarter turn with probability 2*|phi|/pi -- unbiased in
# expectation for an ISOLATED gate. cx/cy/cz/swap/iswap/h/x/y/z/s/sdg/
# sx/sxdg are all exact stabilizer operations with zero error (verified:
# 0/300 mismatches against an exact reference for a Bell-pair cx).
#
# t = rz(pi/4) and tdg = rz(-pi/4) sit exactly at the quadrant boundary,
# giving p = 0.5 always -- a fixed, uniform, closed-form case. Generic
# rz(theta) needs a per-instance mechanism, since Aer's noise dispatch
# can't carry a continuous per-instance angle.

import math
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit_aer.noise import NoiseModel, QuantumError

HALF_PI = math.pi / 2


def reduce_to_quadrant(theta):
    """Reduce an arbitrary angle to (residual, n_s): residual in
    [-pi/4, pi/4], n_s = the signed count of EXACT (zero-cost) quarter
    turns already stripped by QStabilizer::RZ's own deterministic
    while-loops, before the stochastic step ever runs."""
    angle = float(theta)
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    n_s = 0
    while (2 * angle) > HALF_PI:
        n_s += 1
        angle -= HALF_PI
    while (2 * angle) < -HALF_PI:
        n_s -= 1
        angle += HALF_PI
    return angle, n_s


def rz_error_probability(theta):
    """Exact probability of the stochastic rounding step inserting an
    extra quarter turn, for an ISOLATED rz(theta) -- i.e. the first (or
    only) magic gate touching this qubit since its last reset or
    measurement. Not valid for a second rz on the same qubit with no
    intervening reset/measurement: QStabilizer's buffer combines
    successive calls, which this single-gate model does not capture."""
    phi, _ = reduce_to_quadrant(theta)
    return 2 * abs(phi) / math.pi


def _rz_matrix(a):
    return np.array([[np.exp(-1j * a / 2), 0], [0, np.exp(1j * a / 2)]])


def rz_weak_sim_error(theta):
    """Exact single-gate error channel for an isolated rz(theta): compose
    AFTER the ideal rz(theta) (Aer's convention -- ideal gate, then
    error) to correct it into what QrackStabilizer's weak simulation
    actually applies. A 2-branch mixed-unitary channel, not an
    approximation -- exact given the single-gate assumption above."""
    phi, _ = reduce_to_quadrant(theta)
    p = 2 * abs(phi) / math.pi
    sign = 1.0 if phi >= 0 else -1.0

    branch_noop = _rz_matrix(-phi)                  # prob 1-p: nothing physically applied
    branch_kick = _rz_matrix(sign * HALF_PI - phi)   # prob p: extra S/Sdg applied

    return QuantumError([
        (UnitaryGate(branch_noop), 1 - p),
        (UnitaryGate(branch_kick), p),
    ])


# T = RZ(pi/4), Tdg = RZ(-pi/4): both sit exactly at the quadrant
# boundary, giving p = 0.5 exactly -- fixed, uniform across every qubit,
# every instance, no per-instance handling needed.
T_ERROR_PROBABILITY = 0.5
TDG_ERROR_PROBABILITY = 0.5


def t_weak_sim_error():
    return rz_weak_sim_error(math.pi / 4)


def tdg_weak_sim_error():
    return rz_weak_sim_error(-math.pi / 4)


def create_noise_model(n_qubits):
    """Aer NoiseModel for t/tdg only -- fixed angle, so uniform gate-name
    dispatch is exact and sufficient. Every qubit gets the same error
    (uniform profile, full connectivity assumed for QrackStabilizer).

    IMPORTANT: build/run with optimization_level=0. Standard transpiler
    optimization will otherwise remove a t/tdg gate it judges redundant
    (e.g. immediately before measurement with no dependent phase-visible
    operation) -- it has no way to know a noise model depends on that
    gate instance remaining in the circuit, verified directly: a forced
    error silently vanished under default optimization and only applied
    correctly at optimization_level=0.

    Generic rz(theta) is NOT included here -- its error depends
    continuously on theta, which per-(gate name, qubit) noise dispatch
    cannot express, and per-instance labels do not survive transpile().
    Use insert_rz_errors(circuit) for rz instead.
    """
    nm = NoiseModel()
    t_err = t_weak_sim_error()
    tdg_err = tdg_weak_sim_error()
    for q in range(n_qubits):
        nm.add_quantum_error(t_err, "t", [q])
        nm.add_quantum_error(tdg_err, "tdg", [q])
    return nm


def insert_rz_errors(circuit):
    """Return a NEW circuit with the exact weak-simulation error channel
    inserted directly after every rz(theta) instruction, using that
    instance's own angle. Bypasses Aer's gate-name/label noise dispatch
    entirely (labels do not survive transpile(); gate-name dispatch
    can't carry a per-instance parameter) -- verified this direct
    insertion is robust where both alternatives failed.

    The inserted errors must still not be optimized away afterward:
    build/run with optimization_level=0, same as create_noise_model().

    Scope, same caveat as rz_error_probability(): only exact for a
    qubit's first magic gate since its last reset/measurement.
    """
    new_qc = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for inst in circuit.data:
        new_qc.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name == "rz":
            theta = float(inst.operation.params[0])
            err = rz_weak_sim_error(theta)
            new_qc.append(err.to_instruction(), inst.qubits)
    return new_qc
