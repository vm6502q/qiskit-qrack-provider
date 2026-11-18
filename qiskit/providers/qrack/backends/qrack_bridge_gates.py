# (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
#
# Qiskit gate primitives for QrackBridge couplers that Qiskit's standard library lacks.
#
# By (Anthropic) Claude and Dan Strano
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

"""FSimGate and ISwapDgGate, matching QrackBridge.fsim() and QrackBridge.adjiswap()."""

import math

import numpy as np

from qiskit.circuit import Gate, QuantumCircuit, Parameter
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library import CPhaseGate, XXPlusYYGate, iSwapGate


class FSimGate(Gate):
    r"""Qrack's (and Google's) fermionic-simulation gate.

    .. math::

        \mathrm{fSim}(\theta, \phi) = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & \cos\theta & -i\sin\theta & 0 \\
            0 & -i\sin\theta & \cos\theta & 0 \\
            0 & 0 & 0 & e^{-i\phi}
        \end{pmatrix}

    Symmetric in its two qubits. fSim(-pi/2, 0) = iSWAP, fSim(pi/2, 0) = iSWAP^dagger,
    fSim(0, pi) = CZ.
    """

    def __init__(self, theta, phi, label=None):
        super().__init__("fsim", 2, [theta, phi], label=label)

    def _define(self):
        theta, phi = self.params
        qc = QuantumCircuit(2, name=self.name)
        qc.append(XXPlusYYGate(2 * theta, 0), [0, 1])
        qc.append(CPhaseGate(-phi), [0, 1])
        self.definition = qc

    def inverse(self, annotated=False):
        theta, phi = self.params
        return FSimGate(-theta, -phi)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta, phi = (float(p) for p in self.params)
        c, s = math.cos(theta), math.sin(theta)
        m = np.array(
            [[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, np.exp(-1j * phi)]],
            dtype=dtype or complex,
        )
        return m


class ISwapDgGate(Gate):
    r"""Adjoint of iSWAP (QrackBridge.adjiswap()). Named "iswap_dg", like iSwapGate().inverse()."""

    def __init__(self, label=None):
        super().__init__("iswap_dg", 2, [], label=label)

    def _define(self):
        qc = QuantumCircuit(2, name=self.name)
        qc.append(XXPlusYYGate(math.pi, 0), [0, 1])
        self.definition = qc

    def inverse(self, annotated=False):
        return iSwapGate()

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return np.array(
            [[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]], dtype=dtype or complex
        )


def _register_equivalences(lib=SessionEquivalenceLibrary):
    """Let BasisTranslator lower fsim/iswap_dg to other targets (and raise iswap to them)."""
    th, ph = Parameter("theta"), Parameter("phi")
    gate = FSimGate(th, ph)
    gate._define()
    lib.add_equivalence(FSimGate(th, ph), gate.definition)
    gate = ISwapDgGate()
    gate._define()
    lib.add_equivalence(ISwapDgGate(), gate.definition)
    qc = QuantumCircuit(2)
    qc.append(ISwapDgGate(), [0, 1])
    lib.add_equivalence(iSwapGate().inverse(), qc)


_register_equivalences()
