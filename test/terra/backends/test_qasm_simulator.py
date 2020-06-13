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
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsTGate
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsCCXGate
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsCGates
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsMinimalBasis
from test.terra.backends.qasm_simulator.qasm_unitary_gate import QasmUnitaryGateTests
from test.terra.backends.qasm_simulator.qasm_unitary_gate import QasmDiagonalGateTests
from test.terra.backends.qasm_simulator.qasm_initialize import QasmInitializeTests
from test.terra.backends.qasm_simulator.qasm_multiplexer import QasmMultiplexerTests
# Conditional instruction tests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalGateTests
#from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalUnitaryTests
# Algorithm circuit tests
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTests
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTestsMinimalBasis
# Other tests
from test.terra.backends.qasm_simulator.qasm_thread_management import QasmThreadManagementTests
from test.terra.backends.qasm_simulator.qasm_fusion import QasmFusionTests
from test.terra.backends.qasm_simulator.qasm_delay_measure import QasmDelayMeasureTests
from test.terra.backends.qasm_simulator.qasm_basics import QasmBasicsTests


class TestQasmSimulator(common.QiskitAerTestCase,
                        QasmMeasureTests,
                        QasmMultiQubitMeasureTests,
                        QasmResetTests,
                        QasmInitializeTests,
                        QasmConditionalGateTests,
                        #QasmConditionalUnitaryTests,
                        QasmCliffordTests,
                        QasmCliffordTestsWaltzBasis,
                        QasmCliffordTestsMinimalBasis,
                        QasmNonCliffordTestsTGate,
                        QasmNonCliffordTestsCCXGate,
                        QasmNonCliffordTestsCGates,
                        QasmNonCliffordTestsWaltzBasis,
                        QasmNonCliffordTestsMinimalBasis,
                        QasmMultiplexerTests,
                        QasmAlgorithmTests,
                        QasmAlgorithmTestsWaltzBasis,
                        QasmAlgorithmTestsMinimalBasis,
                        QasmUnitaryGateTests,
                        QasmDiagonalGateTests,
                        QasmThreadManagementTests,
                        QasmFusionTests,
                        QasmDelayMeasureTests,
                        QasmBasicsTests):
    """QasmSimulator automatic method tests."""

    BACKEND_OPTS = {
        "seed_simulator": 2113
    }


if __name__ == '__main__':
    unittest.main()
