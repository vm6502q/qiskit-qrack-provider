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

import unittest
from test.terra import common

# Basic circuit instruction tests
from test.terra.backends.qasm_simulator.qasm_reset import QasmResetTests
from test.terra.backends.qasm_simulator.qasm_measure import QasmMeasureTests
from test.terra.backends.qasm_simulator.qasm_measure import QasmMultiQubitMeasureTests
from test.terra.backends.qasm_simulator.qasm_cliffords import QasmCliffordTests
from test.terra.backends.qasm_simulator.qasm_cliffords import QasmCliffordTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_cliffords import QasmCliffordTestsMinimalBasis
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTests
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsMinimalBasis
from test.terra.backends.qasm_simulator.qasm_unitary_gate import QasmUnitaryGateTests
# Conditional instruction tests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalGateTests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalUnitaryTests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalKrausTests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalSuperOpTests
# Algorithm circuit tests
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTests
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTestsMinimalBasis
# Other tests
from test.terra.backends.qasm_simulator.qasm_method import QasmMethodTests


class TestQasmDensityMatrixSimulator(common.QiskitAerTestCase,
                                     QasmMethodTests,
                                     QasmMeasureTests,
                                     QasmMultiQubitMeasureTests,
                                     QasmResetTests,
                                     QasmConditionalGateTests,
                                     QasmConditionalUnitaryTests,
                                     QasmConditionalKrausTests,
                                     QasmConditionalSuperOpTests,
                                     QasmCliffordTests,
                                     QasmCliffordTestsWaltzBasis,
                                     QasmCliffordTestsMinimalBasis,
                                     QasmNonCliffordTests,
                                     QasmNonCliffordTestsWaltzBasis,
                                     QasmNonCliffordTestsMinimalBasis,
                                     QasmAlgorithmTests,
                                     QasmAlgorithmTestsWaltzBasis,
                                     QasmAlgorithmTestsMinimalBasis,
                                     QasmUnitaryGateTests):
    """QasmSimulator density_matrix method tests."""

    BACKEND_OPTS = {
        #"seed_simulator": 314159,
        #"method": "density_matrix"
    }


if __name__ == '__main__':
    unittest.main()
