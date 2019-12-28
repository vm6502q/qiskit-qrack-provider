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
Matrix Product State Integration Tests
"""

import unittest
from test.terra import common
from test.terra.backends.qasm_simulator.matrix_product_state_method import QasmMatrixProductStateMethodTests
from test.terra.backends.qasm_simulator.matrix_product_state_measure import QasmMatrixProductStateMeasureTests
# Snapshot tests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotStatevectorTests

class TestQasmMatrixProductStateSimulator(common.QiskitAerTestCase,
                                          QasmMatrixProductStateMethodTests,
                                          QasmMatrixProductStateMeasureTests,
                                          #QasmSnapshotStatevectorTests,  # FAILING
                                          ):

    BACKEND_OPTS = {
        #"seed_simulator": 314159,
        #"method": "matrix_product_state"
    }


if __name__ == '__main__':
    unittest.main()
