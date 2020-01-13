# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# NOTICE: Daniel Strano, one of the authors of vm6502q/qrack, has modified
# files in this directory to use the Qrack provider instead of the
# Aer provider, for the Qrack provider's own coverage.
"""
QasmSimulator Integration Tests
"""

from test.terra.reference import ref_2q_clifford
from test.terra.reference import ref_non_clifford
from qiskit.compiler import assemble
from qiskit.providers.qrack import QasmSimulator


class QasmMethodTests:
    """QasmSimulator method option tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test Clifford circuits with clifford and non-clifford noise
    # ---------------------------------------------------------------------
    def test_backend_method_clifford_circuits(self):
        """Test statevector method is used for Clifford circuit"""
        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)

        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        success = getattr(result, 'success', False)
        self.assertTrue(success)
        # Check simulation method
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method != 'automatic':
            self.compare_result_metadata(result, circuits, 'method', method)
        else:
            self.compare_result_metadata(result, circuits, 'method',
                                         'stabilizer')

    # ---------------------------------------------------------------------
    # Test non-Clifford circuits with clifford and non-clifford noise
    # ---------------------------------------------------------------------
    def test_backend_method_nonclifford_circuits(self):
        """Test statevector method is used for Clifford circuit"""
        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)

        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        success = getattr(result, 'success', False)
        # Check simulation method
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method == 'stabilizer':
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            if method == 'automatic':
                target_method = 'statevector'
            else:
                target_method = method
            self.compare_result_metadata(result, circuits, 'method',
                                         target_method)
