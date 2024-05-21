# This code is based on and adapted from https://github.com/Qiskit/qiskit-aer/blob/master/qiskit/providers/aer/aerprovider.py
#
# Adapted by Daniel Strano
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name, bad-continuation

"""Provider for Qrack backends."""

from qiskit.providers.providerutils import filter_backends

from .backends import QasmSimulator


class QrackProvider:
    """Provider for Qrack backends."""

    def __init__(self):
        super().__init__()

        # Populate the list of Qrack simulator providers.
        self._backends = [
            QasmSimulator(configuration=None, provider=self)
        ]

    def backends(self, name=None, filters=None, **kwargs):
        # pylint: disable=arguments-differ
        backends = self._backends
        if name:
            backends = [backend for backend in backends if backend.name() == name]

        return filter_backends(backends, filters=filters, **kwargs)

    def __str__(self):
        return 'QrackProvider'
