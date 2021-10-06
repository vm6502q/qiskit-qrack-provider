"""
This module implements the job class used by simulator backends.
Taken mostly from https://github.com/Qiskit/qiskit-qcgpu-provider/blob/master/qiskit_qcgpu_provider/job.py
"""


from qiskit.providers.job import JobV1
from qiskit.providers import JobStatus


class QrackJob(JobV1):
    """
    QrackJob class.
    This is a mocking futures class, used only 
    to fit the API.
    """

    def __init__(self, backend, job_id, val, qobj):
        super().__init__(backend, job_id)
        self._val = val
        self._qobj = qobj
        self._future = None

    def submit(self):
        """
        Submit the job to the backend for execution.
        Raises:
            QobjValidationError: if the JSON serialization of the Qobj passed
            during construction does not validate against the Qobj schema.
            JobError: if trying to re-submit the job.
        # """
        return

    def result(self, timeout=None):
        # pylint: disable=arguments-differ
        """
        Get job result. The behavior is the same as the underlying
        concurrent Future objects,
        https://docs.python.org/3/library/concurrent.futures.html#future-objects
        Args:
            timeout (float): number of seconds to wait for results.
        Returns:
            qiskit.Result: Result object
        Raises:
            concurrent.futures.TimeoutError: if timeout occurred.
            concurrent.futures.CancelledError: if job cancelled before completed.
        """
        return self._val

    def cancel(self):
        return

    def status(self):
        """
        Gets the status of the job by querying the Python's future
        Returns:
            JobStatus: The current JobStatus
        Raises:
            JobError: If the future is in unexpected state
            concurrent.futures.TimeoutError: if timeout occurred.
        """
        return JobStatus.DONE
