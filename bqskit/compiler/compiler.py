"""Local compiler override for PassData persistence."""
from __future__ import annotations

import atexit
import functools
import logging
import pickle
import signal
import subprocess
import sys
import time
import uuid
from multiprocessing.connection import Client
from multiprocessing.connection import Connection
from subprocess import Popen
from types import FrameType
from typing import Literal
from typing import MutableMapping
from typing import overload
from typing import TYPE_CHECKING

from bqskit.compiler.status import CompilationStatus
from bqskit.compiler.task import CompilationTask
from bqskit.compiler.workflow import Workflow
from bqskit.compiler.workflow import WorkflowLike
from bqskit.runtime import default_server_port
from bqskit.runtime import default_worker_port
from bqskit.runtime.message import RuntimeMessage

if TYPE_CHECKING:
    from typing import Any
    from bqskit.ir.circuit import Circuit
    from bqskit.compiler.passdata import PassData

_logger = logging.getLogger(__name__)


class Compiler:
    """A compiler is responsible for accepting and managing compilation tasks."""

    def __init__(
        self,
        ip: str | None = None,
        port: int = default_server_port,
        num_workers: int = -1,
        runtime_log_level: int = logging.WARNING,
        worker_port: int = default_worker_port,
        num_blas_threads: int = 1,
    ) -> None:
        self.p: Popen | None = None  # type: ignore
        self.conn: Connection | None = None

        _compiler_instances.add(self)

        if ip is None:
            ip = 'localhost'
            self._start_server(
                num_workers,
                runtime_log_level,
                worker_port,
                num_blas_threads,
            )

        self._connect_to_server(ip, port, self.p is not None)

    def _start_server(
        self,
        num_workers: int,
        runtime_log_level: int,
        worker_port: int,
        num_blas_threads: int,
    ) -> None:
        params = f'{num_workers}, '
        params += f'log_level={runtime_log_level}, '
        params += f'{worker_port=}, '
        params += f'{num_blas_threads=}, '
        import_str = 'from bqskit.runtime.attached import start_attached_server'
        launch_str = f'{import_str}; start_attached_server({params})'
        if sys.platform == 'win32':
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            flags = 0
        self.p = Popen([sys.executable, '-c', launch_str], creationflags=flags)
        _logger.debug('Starting runtime server process.')

    def _connect_to_server(self, ip: str, port: int, attached: bool) -> None:
        max_retries = 8
        wait_time = .25
        current_retry = 0
        while current_retry < max_retries or attached:
            try:
                family = 'AF_INET' if sys.platform == 'win32' else None
                conn = Client((ip, port), family)
            except ConnectionRefusedError:
                if wait_time > 4:
                    _logger.warning(
                        'Connection refused by runtime server.'
                        ' Retrying in %s seconds.', wait_time,
                    )
                if wait_time > 16 and attached:
                    _logger.warning(
                        'Connection is still refused by runtime server.'
                        ' This may be due to the server not being started.'
                        ' You may want to check the server logs, by starting'
                        ' the compiler with "runtime_log_level" set. You'
                        ' can also try launching the bqskit runtime in'
                        ' detached mode. See the bqskit runtime documentation'
                        ' for more information:'
                        ' https://bqskit.readthedocs.io/en/latest/guides/'
                        'distributing.html',
                    )
                time.sleep(wait_time)
                wait_time *= 2
                current_retry += 1
            else:
                self.conn = conn
                handle = functools.partial(sigint_handler, compiler=self)
                self.old_signal = signal.signal(signal.SIGINT, handle)
                if self.conn is None:
                    raise RuntimeError('Connection unexpectedly none.')
                msg, payload = self._send_recv(RuntimeMessage.CONNECT, sys.path)
                if msg != RuntimeMessage.READY:
                    raise RuntimeError(f'Unexpected message type: {msg}.')
                _logger.debug('Successfully connected to runtime server.')
                return
        raise RuntimeError('Client connection refused')

    def __enter__(self) -> Compiler:
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.send((RuntimeMessage.DISCONNECT, None))
                try:
                    self.conn.recv()
                except EOFError:
                    pass
                self.conn.close()
            except Exception as e:
                _logger.debug(
                    'Unsuccessfully disconnected from runtime server.',
                )
                _logger.debug(e)
            else:
                _logger.debug('Disconnected from runtime server.')
            finally:
                self.conn = None

        if self.p is not None and self.p.pid is not None:
            try:
                if sys.platform == 'win32':
                    self.p.send_signal(signal.CTRL_C_EVENT)
                else:
                    self.p.send_signal(signal.SIGINT)
                _logger.debug('Interrupting attached runtime server.')
                self.p.communicate(timeout=1)

            except subprocess.TimeoutExpired:
                self.p.kill()
                _logger.debug('Killing attached runtime server.')
                try:
                    self.p.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    _logger.warning(
                        'Failed to kill attached runtime server.'
                        ' It may still be running as a zombie process.',
                    )
                else:
                    _logger.debug('Attached runtime server is down.')

            except Exception as e:
                _logger.warning(
                    f'Error while shuting down attached runtime server: {e}.',
                )

            else:
                _logger.debug('Successfully shutdown attached runtime server.')

            finally:
                self.p = None

        if hasattr(self, 'old_signal'):
            signal.signal(signal.SIGINT, self.old_signal)
            del self.old_signal

        _compiler_instances.discard(self)
        _logger.debug('Compiler has been closed.')

    def submit(
        self,
        circuit: Circuit,
        workflow: WorkflowLike,
        request_data: bool = False,
        logging_level: int | None = None,
        max_logging_depth: int = -1,
        data: MutableMapping[str, Any] | None = None,
    ) -> uuid.UUID:
        task = CompilationTask(circuit, Workflow(workflow))

        task.request_data = request_data
        task.logging_level = logging_level or self._discover_lowest_log_level()
        task.max_logging_depth = max_logging_depth
        if data is not None:
            task.data.update(data)

        self._send(RuntimeMessage.SUBMIT, task)
        return task.task_id

    def status(self, task_id: uuid.UUID) -> CompilationStatus:
        msg, payload = self._send_recv(RuntimeMessage.STATUS, task_id)
        if msg != RuntimeMessage.STATUS:
            raise RuntimeError(f'Unexpected message type: {msg}.')
        return payload

    def result(self, task_id: uuid.UUID) -> Circuit | tuple[Circuit, PassData]:
        msg, payload = self._send_recv(RuntimeMessage.REQUEST, task_id)
        if msg != RuntimeMessage.RESULT:
            raise RuntimeError(f'Unexpected message type: {msg}.')
        return payload

    def cancel(self, task_id: uuid.UUID) -> bool:
        msg, _ = self._send_recv(RuntimeMessage.CANCEL, task_id)
        if msg != RuntimeMessage.CANCEL:
            raise RuntimeError(f'Unexpected message type: {msg}.')
        return True

    @overload
    def compile(
        self,
        circuit: Circuit,
        workflow: WorkflowLike,
        request_data: Literal[False] = ...,
        logging_level: int | None = ...,
        max_logging_depth: int = ...,
        data: MutableMapping[str, Any] | None = ...,
    ) -> Circuit:
        ...

    @overload
    def compile(
        self,
        circuit: Circuit,
        workflow: WorkflowLike,
        request_data: Literal[True],
        logging_level: int | None = ...,
        max_logging_depth: int = ...,
        data: MutableMapping[str, Any] | None = ...,
    ) -> tuple[Circuit, PassData]:
        ...

    @overload
    def compile(
        self,
        circuit: Circuit,
        workflow: WorkflowLike,
        request_data: bool,
        logging_level: int | None = ...,
        max_logging_depth: int = ...,
        data: MutableMapping[str, Any] | None = ...,
    ) -> Circuit | tuple[Circuit, PassData]:
        ...

    def compile(
        self,
        circuit: Circuit,
        workflow: WorkflowLike,
        request_data: bool = False,
        logging_level: int | None = None,
        max_logging_depth: int = -1,
        data: MutableMapping[str, Any] | None = None,
    ) -> Circuit | tuple[Circuit, PassData]:
        """
        Submit a task, wait for its results; see :func:`submit` for more.

        If the caller provides a mutable `data` mapping, request task data from
        the runtime and merge it back into that object before returning.
        """
        persist_data = data is not None
        effective_request_data = request_data or persist_data

        task_id = self.submit(
            circuit,
            workflow,
            effective_request_data,
            logging_level,
            max_logging_depth,
            data,
        )
        result = self.result(task_id)

        if persist_data:
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError(
                    'Expected compiler to return circuit and pass data.',
                )

            compiled_circuit, result_data = result
            if hasattr(data, 'update') and callable(getattr(data, 'update')):
                data.update(result_data)
            elif hasattr(data, 'become') and callable(getattr(data, 'become')):
                data.become(result_data)
            elif hasattr(data, 'clear') and callable(getattr(data, 'clear')):
                data.clear()
                data.update(result_data)
            else:
                raise TypeError(
                    'Expected mutable data object with either `update`, '
                    '`become`, or `clear`/`update` support.',
                )

            if not request_data:
                result = compiled_circuit

        time.sleep(0.05 if self.p is not None else 0.5)
        self._recv_log_error_until_empty()

        return result

    def _send(self, msg: RuntimeMessage, payload: Any) -> None:
        if self.conn is None:
            raise RuntimeError('Connection unexpectedly none.')

        try:
            self._recv_log_error_until_empty()
            self.conn.send((msg, payload))

        except Exception as e:
            self.conn = None
            self.close()
            if isinstance(e, (EOFError, ConnectionResetError)):
                raise RuntimeError('Server connection unexpectedly closed.')
            else:
                raise RuntimeError(
                    'Server connection unexpectedly closed.',
                ) from e

    def _send_recv(
        self,
        msg: RuntimeMessage,
        payload: Any,
    ) -> tuple[RuntimeMessage, Any]:
        if self.conn is None:
            raise RuntimeError('Connection unexpectedly none.')

        try:
            self._recv_log_error_until_empty()
            self.conn.send((msg, payload))
            return self._recv_handle_log_error()

        except Exception as e:
            self.conn = None
            self.close()
            raise RuntimeError('Server connection unexpectedly closed.') from e

    def _recv_handle_log_error(self) -> tuple[RuntimeMessage, Any]:
        if self.conn is None:
            raise RuntimeError('Connection unexpectedly none.')

        to_return = None
        while to_return is None or self.conn.poll():
            msg, payload = self.conn.recv()

            if msg == RuntimeMessage.LOG:
                record = pickle.loads(payload)
                if isinstance(record, logging.LogRecord):
                    logger = logging.getLogger(record.name)
                    if logger.isEnabledFor(record.levelno):
                        logger.handle(record)
                else:
                    name, levelno, msg = record
                    logger = logging.getLogger(name)
                    logger.log(levelno, msg)

            elif msg == RuntimeMessage.ERROR:
                raise RuntimeError(payload)

            else:
                to_return = (msg, payload)

        return to_return

    def _recv_log_error_until_empty(self) -> None:
        if self.conn is None:
            raise RuntimeError('Connection unexpectedly none.')

        while self.conn.poll():
            msg, payload = self.conn.recv()

            if msg == RuntimeMessage.LOG:
                logger = logging.getLogger(payload.name)
                if logger.isEnabledFor(payload.levelno):
                    logger.handle(payload)

            elif msg == RuntimeMessage.ERROR:
                raise RuntimeError(payload)

            else:
                raise RuntimeError(f'Unexpected message type: {msg}.')

    def _discover_lowest_log_level(self) -> int:
        lowest_level_found_so_far = logging.getLogger().getEffectiveLevel()

        for _, logger in logging.getLogger().manager.loggerDict.items():
            if isinstance(logger, logging.PlaceHolder):
                continue

            if logger.getEffectiveLevel() < lowest_level_found_so_far:
                lowest_level_found_so_far = logger.getEffectiveLevel()

        return lowest_level_found_so_far


def sigint_handler(signum: int, frame: FrameType, compiler: Compiler) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _logger.critical('Compiler interrupted.')
    compiler.close()
    raise KeyboardInterrupt


_compiler_instances: set[Compiler] = set()


@atexit.register
def _cleanup_compiler_instances() -> None:
    for compiler in list(_compiler_instances):
        compiler.close()
