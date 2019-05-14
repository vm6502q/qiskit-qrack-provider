# This code is based on and adapted from https://github.com/Qiskit/qiskit-qcgpu-provider/blob/master/qiskit_qcgpu_provider/qasm_simulator.py
#
# Adapted by Daniel Strano. Many thanks to Adam Kelley for pioneering a third-party Qiskit provider.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name


import uuid
import time
import numpy as np
from collections import Counter
import logging
import os
from math import log2
from qiskit.util import local_hardware_info
from qiskit.providers.models import BackendConfiguration
from .qrack_controller_wrapper import PyQrackController, qrack_controller_factory
from ..qrackerror import QrackError
from ..version import __version__

from ..qrackjob import QrackJob
from ..qrackerror import QrackError

from qiskit.providers import BaseBackend
from qiskit.result import Result

logger = logging.getLogger(__name__)


class QasmSimulator(BaseBackend):
    """Contains an OpenCL based backend"""

    MAX_QUBITS_MEMORY = 30  # Should do something smarter here,
    # but needs to be implemented on the Qrack side.
    # Will also depend on choosing the optimal backend.

    DEFAULT_CONFIGURATION = {
        'backend_name': 'qasm_simulator',
        'backend_version': __version__,
        'n_qubits': MAX_QUBITS_MEMORY,
        'url': 'https://github.com/vm6502q/qiskit-aer',
        'simulator': True,
        'local': True,
        'conditional': False,
        'open_pulse': False,
        'memory': True,
        'max_shots': 65536,
        'description': 'An OpenCL based qasm simulator',
        'basis_gates': ['u1', 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z',
                        'h', 's', 'sdg', 't', 'tdg', 'ccx', 'swap', 'reset'],
        'gates': [{'name': 'u1', 'parameters': ['lambda'], 'qasm_def': 'gate u1(lambda) q { U(0,0,lambda) q; }'},
                  {'name': 'u2', 'parameters': ['phi', 'lambda'], 'qasm_def': 'gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }'},
                  {'name': 'u3', 'parameters': ['theta', 'phi', 'lambda'], 'qasm_def': 'gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }'},
                  {'name': 'cx', 'parameters': ['c', 't'], 'qasm_def': 'gate cx c,t { CX c,t; }'},
                  {'name': 'cz', 'parameters': ['c', 't'], 'qasm_def': 'gate cz c,t { CZ c,t; }'},
                  {'name': 'id', 'parameters': ['a'], 'qasm_def': 'gate id a { U(0,0,0) a; }'},
                  {'name': 'x', 'parameters': ['a'], 'qasm_def': 'gate x a { u3(pi,0,pi) a; }'},
                  {'name': 'y', 'parameters': ['a'], 'qasm_def': 'gate y a { u3(pi,pi/2,pi/2) a; }'},
                  {'name': 'z', 'parameters': ['z'], 'qasm_def': 'gate z a { u1(pi) a; }'},
                  {'name': 'h', 'parameters': ['a'], 'qasm_def': 'gate h a { u2(0,pi) a; }'},
                  {'name': 's', 'parameters': ['a'], 'qasm_def': 'gate s a { u1(pi/2) a; }'},
                  {'name': 'sdg', 'parameters': ['a'], 'qasm_def': 'gate s a { u1(-pi/2) a; }'},
                  {'name': 't', 'parameters': ['a'], 'qasm_def': 'gate t a { u1(pi/4) a; }'},
                  {'name': 'tdg', 'parameters': ['a'], 'qasm_def': 'gate t a { u1(-pi/4) a; }'}
                  #TODO: ccx, swap, reset
                 ]}

    def __init__(self, configuration=None, provider=None):
        configuration = configuration or BackendConfiguration.from_dict(
            self.DEFAULT_CONFIGURATION)
        super().__init__(configuration=configuration, provider=provider)

        self._configuration = configuration
        self._number_of_qubits = None
        self._number_of_cbits = None
        self._statevector = None
        self._results = {}
        self._shots = {}
        self._local_random = np.random.RandomState()
        self._sample_measure = False
        self._chop_threshold = 15  # chop to 10^-15

    #@profile
    def run(self, qobj):
        """Run qobj asynchronously.
        Args:
            qobj (Qobj): payload of the experiment
        Returns:
            QrackJob: derived from BaseJob
        """

        job_id = str(uuid.uuid4())
        job = QrackJob(self, job_id, self._run_job(job_id, qobj), qobj)
        return job
    #@profile
    def _run_job(self, job_id, qobj):
        """Run experiments in qobj
        Args:
            job_id (str): unique id for the job.
            qobj (Qobj): job description
        Returns:
            Result: Result object
        """
        self._shots = qobj.config.shots
        self._memory = qobj.config.memory
        self._qobj_config = qobj.config
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
            'header': qobj.header.as_dict()
        }

        return Result.from_dict(result)

    #@profile
    def run_experiment(self, experiment):
        """Run an experiment (circuit) and return a single experiment result.
        Args:
            experiment (QobjExperiment): experiment from qobj experiments list
        Returns:
            dict: A dictionary of results.
            dict: A result dictionary
        Raises:
            QrackSimulatorError: If the number of qubits is too large, or another
                error occurs during execution.
        """
        self._number_of_qubits = experiment.header.n_qubits
        self._number_of_cbits = experiment.header.memory_slots
        self._classical_state = 0
        self._statevector = 0

        if hasattr(experiment.config, 'seed'):
            seed = experiment.config.seed
        elif hasattr(self._qobj_config, 'seed'):
            seed = self._qobj_config.seed
        else:
            # For compatibility on Windows force dyte to be int32
            # and set the maximum value to be (2 ** 31) - 1
            seed = np.random.randint(2147483647, dtype='int32')
        self._local_random.seed(seed)

        self._can_sample(experiment)

        if not self._sample_measure:
            raise QrackSimulatorError('Measurements are only supported at the end')

        experiment = experiment.as_dict()

        samples = []

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
            elif name == 'swap':
                sim.swap(operation['qubits'][0], operation['qubits'][1])
            elif name == 'reset':
                sim.reset(operation['qubits'][0])
            elif name == 'measure':
                samples.append((operation['qubits'][0], operation['memory'][0]))

        if self._number_of_cbits > 0:
            memory = self._add_sample_measure(samples, sim, self._shots)
        else:
            memory = []

        end = time.time()

        # amps = sim.amplitudes().round(self._chop_threshold)
        # amps = np.stack((amps.real, amps.imag), axis=-1)

        data = {'counts': dict(Counter(memory))}

        if self._memory:
            data['memory'] = memory

        return {
            'name': experiment['header']['name'],
            'shots': self._shots,
            'data': data,
            'seed': seed,
            'status': 'DONE',
            'success': True,
            'time_taken': (end - start),
            'header': experiment['header']
        }

    #@profile
    def _add_sample_measure(self, measure_params, sim, num_samples):
        """Generate memory samples from current statevector.
        Taken almost straight from the terra source code.
        Args:
            measure_params (list): List of (qubit, clbit) values for
                                   measure instructions to sample.
            num_samples (int): The number of memory samples to generate.
        Returns:
            list: A list of memory values in hex format.
        """
        probabilities = np.reshape(sim.probabilities(), self._number_of_qubits * [2])

        # Get unique qubits that are actually measured
        measured_qubits = list(set([qubit for qubit, clbit in measure_params]))
        num_measured = len(measured_qubits)


        # Axis for numpy.sum to compute probabilities
        axis = list(range(self._number_of_qubits))

        for qubit in reversed(measured_qubits):
            # Remove from largest qubit to smallest so list position is correct
            # with respect to position from end of the list
            axis.remove(self._number_of_qubits - 1 - qubit)
        
        
        probabilities = np.reshape(np.sum(probabilities,
                                          axis=tuple(axis)),
                                   2 ** num_measured)
        # Generate samples on measured qubits
        samples = self._local_random.choice(range(2 ** num_measured),
                                            num_samples, p=probabilities)
        # Convert to bit-strings
        memory = []
        for sample in samples:
            classical_state = self._classical_state
            for qubit, cbit in measure_params:
                qubit_outcome = int((sample & (1 << qubit)) >> qubit)
                bit = 1 << cbit
                classical_state = (classical_state & (~bit)) | (qubit_outcome << cbit)
            value = bin(classical_state)[2:]
            memory.append(hex(int(value, 2)))
        return memory

    #@profile
    def _can_sample(self, experiment):
        """Determine if sampling can be used for an experiment
        Args:
            experiment (QobjExperiment): a qobj experiment
        """
        measure_flags = {}
        if hasattr(experiment.config, 'allows_measure_sampling'):
            self._sample_measure = experiment.config.allows_measure_sampling
        else:

            for instruction in experiment.instructions:
                if instruction.name == "reset":
                    measure_flags[instruction.qubits[0]] = False
                    self._sample_measure = False
                    return

                if measure_flags.get(instruction.qubits[0], False):
                    if instruction.name not in ["measure", "barrier", "id", "u0"]:
                        for qubit in instruction.qubits:
                            measure_flags[qubit] = False
                            return
                elif instruction.name == "measure":
                     for qubit in instruction.qubits:
                            measure_flags[qubit] = True
        
        self._sample_measure = True

        for key, value in measure_flags.items():
            if value == False:
                self._sample_measure = False
                return

    @staticmethod
    def name():
        return 'qasm_simulator'
