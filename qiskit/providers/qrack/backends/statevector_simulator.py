# This code is based on and adapted from https://github.com/Qiskit/qiskit-qcgpu-provider/blob/master/qiskit_qcgpu_provider/statevector_simulator.py
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

# pylint: disable=invalid-name

"""
Qrack Statevector simulator backend.
"""

import uuid
import time
import numpy as np
import logging
import os
from math import log2
from qiskit.util import local_hardware_info
from qiskit.providers.models import BackendConfiguration
from .qrack_controller_wrapper import qrack_controller_factory
from ..qrackerror import QrackError
from ..version import __version__

from ..qrackjob import QrackJob
from ..qrackerror import QrackError

from qiskit.providers import BaseBackend
from qiskit.result import Result

# Logger
logger = logging.getLogger(__name__)


class StatevectorSimulator(BaseBackend):
    """Qrack statevector simulator

    Backend options:

        The following backend options may be used with in the
        `backend_options` kwarg diction for `StatevectorSimulator.run` or
        `qiskit.execute`

        * "zero_threshold" (double): Sets the threshold for truncating
            small values to zero in the result data (Default: 1e-8 for float Qrack and ~1e-18 for double Qrack).

        * "max_memory_mb" (int): Sets the maximum size of memory
            to store a state vector. If a state vector needs more, an error
            is thrown. In general, a state vector of n-qubits uses 2^n complex
            values (16 Bytes). If set to 0, the maximum will be automatically
            set to half the system memory size (Default: 0).
    """

    MAX_QUBIT_MEMORY = int(log2(local_hardware_info()['memory'] * (1024 ** 3) / 16))

    DEFAULT_CONFIGURATION = {
        'backend_name': 'statevector_simulator',
        'backend_version': __version__,
        'n_qubits': MAX_QUBIT_MEMORY,
        'url': 'https://github.com/vm6502q/qiskit-aer',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': 1,
        'description': 'A Qrack-based, GPU-accelerated, C++ statevector simulator for qobj files',
        'coupling_map': None,
        'basis_gates': ['u1', 'u2', 'u3', 'cx', 'cz', 'ch', 'id', 'x', 'y', 'z',
                        'h', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'ccx', 'swap', 'reset'],
        'gates': [{'name': 'u1', 'parameters': ['lambda'], 'qasm_def': 'gate u1(lambda) q { u3(0,0,lambda) q; }'},
                  {'name': 'u2', 'parameters': ['phi', 'lambda'], 'qasm_def': 'gate u2(phi,lambda) q { u3(pi/2,phi,lambda) q; }'},
                  {'name': 'u3', 'parameters': ['theta', 'phi', 'lambda'], 'qasm_def': 'gate u3(theta,phi,lambda) q { u3(theta,phi,lambda) q; }'},
                  {'name': 'cx', 'parameters': ['c', 't'], 'qasm_def': 'gate cx c,t { cx c,t; }'},
                  {'name': 'cz', 'parameters': ['c', 't'], 'qasm_def': 'gate cz c,t { cz c,t; }'},
                  {'name': 'id', 'parameters': ['a'], 'qasm_def': 'gate id a { u3(0,0,0) a; }'},
                  {'name': 'x', 'parameters': ['a'], 'qasm_def': 'gate x a { u3(pi,0,pi) a; }'},
                  {'name': 'y', 'parameters': ['a'], 'qasm_def': 'gate y a { u3(pi,pi/2,pi/2) a; }'},
                  {'name': 'z', 'parameters': ['a'], 'qasm_def': 'gate z a { u1(pi) a; }'},
                  {'name': 'h', 'parameters': ['a'], 'qasm_def': 'gate h a { u2(0,pi) a; }'},
                  {'name': 's', 'parameters': ['a'], 'qasm_def': 'gate s a { u1(pi/2) a; }'},
                  {'name': 'sdg', 'parameters': ['a'], 'qasm_def': 'gate s a { u1(-pi/2) a; }'},
                  {'name': 't', 'parameters': ['a'], 'qasm_def': 'gate t a { u1(pi/4) a; }'},
                  {'name': 'tdg', 'parameters': ['a'], 'qasm_def': 'gate t a { u1(-pi/4) a; }'},
                  {'name': 'rx', 'parameters': ['theta', 'a'], 'qasm_def': 'gate rx(theta) a { u3(theta, -pi/2, pi/2) a; }'},
                  {'name': 'ry', 'parameters': ['theta', 'a'], 'qasm_def': 'gate ry(theta) a { u3(theta, 0, 0) a; }'},
                  {'name': 'rz', 'parameters': ['phi', 'a'], 'qasm_def': 'gate rz(phi) a { u1(phi) a; }'}
                  #TODO: ch, ccx, swap, reset
                 ],
        # Location where we put external libraries that will be loaded at runtime
        # by the simulator extension
        'library_dir': os.path.dirname(__file__)
    }

    def __init__(self, configuration=None, provider=None):
        configuration = configuration or BackendConfiguration.from_dict(
            self.DEFAULT_CONFIGURATION)
        super().__init__(configuration=configuration, provider=provider)

        self._configuration = configuration
        self._number_of_qubits = None
        self._statevector = None
        self._results = {}
        self._chop_threshold = None # chop to 10^-8

    def run(self, qobj):
        """
        Run qobj asynchronously.
        Args:
            qobj (Qobj): payload of the experiment
        Returns:
            QCGPUJob: derived from BaseJob
        """

        job_id = str(uuid.uuid4())
        job = QrackJob(self, job_id, self._run_job(job_id, qobj), qobj)
        return job

    def _run_job(self, job_id, qobj):
        """
        Run experiments in qobj
        Args:
            job_id (str): unique id for the job.
            qobj (Qobj): job description
        Returns:
            Result: Result object
        """
        self._validate(qobj)
        results = []

        start = time.time()
        for experiment in qobj.experiments:
            results.append(self.run_experiment(experiment))
        end = time.time()

        result = {
            'backend_name': self.name(),
            'backend_version': self._configuration.backend_version,
            'qobj_id': qobj.qobj_id,
            'job_id': job_id,
            'results': results,
            'status': 'COMPLETED',
            'success': True,
            'time_taken': (end - start),
            'header': qobj.header.to_dict()
        }

        return Result.from_dict(result) # This can be sped up

    def run_experiment(self, experiment):
        """Run an experiment (circuit) and return a single experiment result.
        Args:
            experiment (QobjExperiment): experiment from qobj experiments list
        Returns:
            dict: A dictionary of results.
            dict: A result dictionary
        Raises:
            QCGPUSimulatorError: If the number of qubits is too large, or another
                error occurs during execution.
        """
        self._number_of_qubits = experiment.header.n_qubits
        self._statevector = 0
        experiment = experiment.to_dict()

        start = time.time()

        try:
            sim = qrack_controller_factory()
            sim.initialize_qreg(self._number_of_qubits)
        except OverflowError:
            raise QrackSimulatorError('too many qubits')

        for operation in experiment['instructions']:
            name = operation['name']

            if name == 'id':
                logger.info('Identity gates are ignored.')
            elif name == 'barrier':
                logger.info('Barrier gates are ignored.')
            elif name == 'u3':
                sim.u(operation['qubits'][0], operation['params'])
            elif name == 'u2':
                sim.u2(operation['qubits'][0], operation['params'])
            elif name == 'u1':
                sim.u1(operation['qubits'][0], operation['params'])
            elif name == 'cx':
                sim.cx(operation['qubits'], 1)
            elif name == 'ccx':
                sim.cx(operation['qubits'], 2)
            elif name == 'cz':
                sim.cz(operation['qubits'], 1)
            elif name == 'ch':
                sim.ch(operation['qubits'], 1)
            elif name == 'h':
                sim.h(operation['qubits'][0])
            elif name == 'x':
                sim.x(operation['qubits'][0])
            elif name == 'y':
                sim.y(operation['qubits'][0])
            elif name == 'z':
                sim.z(operation['qubits'][0])
            elif name == 's':
                sim.s(operation['qubits'][0])
            elif name == 'sdg':
                sim.sdg(operation['qubits'][0])
            elif name == 't':
                sim.t(operation['qubits'][0])
            elif name == 'tdg':
                sim.tdg(operation['qubits'][0])
            elif name == 'rx':
                sim.rx(operation['qubits'][0], operation['params'])
            elif name == 'ry':
                sim.ry(operation['qubits'][0], operation['params'])
            elif name == 'rz':
                sim.rz(operation['qubits'][0], operation['params'])
            elif name == 'swap':
                sim.swap(operation['qubits'][0], operation['qubits'][1])
            elif name == 'reset':
                sim.reset(operation['qubits'][0])

        end = time.time()

        if self._chop_threshold:
            self._statevector = sim.amplitudes().round(self._chop_threshold)
        else:
            self._statevector = sim.amplitudes()
        self._statevector = np.stack((self._statevector.real, self._statevector.imag), axis=-1)
        return {
            'name': experiment['header']['name'],
            'shots': 1,
            'data': {'statevector': self._statevector},
            'status': 'DONE',
            'success': True,
            'time_taken': (end - start),
            'header': experiment['header']
        }

    def _validate(self, qobj, backend_options=None, noise_model=None):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.

        1. Set shots=1.
        2. Check number of qubits will fit in local memory.
        """
        name = self.name()
        if noise_model is not None:
            logger.error("{} cannot be run with a noise.".format(name))
            raise QrackError("{} does not support noise.".format(name))

        n_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if n_qubits > max_qubits:
            raise QrackError('Number of qubits ({}) '.format(n_qubits) +
                           'is greater than maximum ({}) '.format(max_qubits) +
                           'for "{}" '.format(name) +
                           'with {} GB system memory.'.format(int(local_hardware_info()['memory'])))
        if qobj.config.shots != 1:
            logger.info('"%s" only supports 1 shot. Setting shots=1.',
                        name)
            qobj.config.shots = 1
        for experiment in qobj.experiments:
            exp_name = experiment.header.name
            if getattr(experiment.config, 'shots', 1) != 1:
                logger.info('"%s" only supports 1 shot. '
                            'Setting shots=1 for circuit "%s".',
                            name, exp_name)
                experiment.config.shots = 1
