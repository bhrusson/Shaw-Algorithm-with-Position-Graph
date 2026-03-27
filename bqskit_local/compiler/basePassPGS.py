"""This module implements the BasePass abstract base class."""
from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    #from bqskit.compiler.machine import MachineModel
    #from bqskit.compiler.passdata import PassData
    #from bqskit.compiler.workflow import Workflow
    from bqskit.ir.circuit import Circuit
    #from bqskit.qis.graph import CouplingGraph
    from bqskit.qis.state.system import StateSystem
    from bqskit.qis.state.state import StateVector
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

    from bqskit_local.position.state import PositionGraphState
    from bqskit_local.compiler.passDataPGS import PassDataPGS    
    from bqskit_local.compiler.workflowPGS import WorkflowPGS


class BasePassPGS(abc.ABC):
    """
    The abstract base for BQSKit compiler passes.

    All BQSKit algorithms must inherit from BasePass to be run within
    the compiler framework. Each child class will need to implement its
    algorithm inside of the :func:`run` method.

    Examples:
        >>> class PrintCNOTCountPass(BasePass):
        ...     async def run(self, circuit: Circuit, data: PassDataPGS) -> None:
        ...         print(f"Number of CNOTs: {circuit.count(CNOTGate())}")
    """

    @property
    def name(self) -> str:
        """The name of the pass."""
        return self.__class__.__name__

    @abc.abstractmethod
    async def run(self, circuit: Circuit, data: PassDataPGS) -> None:
        """
        Perform the pass's operation on `circuit`.

        Args:
            circuit (Circuit): The circuit to operate on.

            data (PassDataPGS): Associated data for the pass.
                Can be used to get auxillary information from previous
                passes and to store information for future passes.

        Note:
            - This function should be self-contained and have no side effects.
              This is because it will be potentially run in parallel.
        """

    def __str__(self) -> str:
        """Return a string representation of the pass."""
        return self.name

    @staticmethod
    def get_model(_: Any, data: PassDataPGS) -> PositionGraphState:
        """
        Retrieve the machine model from the data dictionary.

        (Deprecated)
        """
        warnings.warn(
            'BasePass calls to retrieve elements from the data dictionary'
            ' are now deprecated. We have upgraded the features of the data'
            ' dictionary. It can now be used directly to retrieve elements'
            " and provide defaults if they don't exist. This warning will"
            ' become an error in the future.',
            DeprecationWarning,
        )
        return data.model

    @staticmethod
    def get_placement(_: Any, data: PassDataPGS) -> list[int]:
        """
        Retrieve the logical to physical qubit map from the data dictionary.

        (Deprecated)
        """
        warnings.warn(
            'BasePass calls to retrieve elements from the data dictionary'
            ' are now deprecated. We have upgraded the features of the data'
            ' dictionary. It can now be used directly to retrieve elements'
            " and provide defaults if they don't exist. This warning will"
            ' become an error in the future.',
            DeprecationWarning,
        )
        return data.placement

    @staticmethod
    def get_connectivity(_: Any, data: PassDataPGS) -> PositionGraphState:
        """
        Retrieve the current connectivity of the circuit.

        (Deprecated)
        """
        warnings.warn(
            'BasePass calls to retrieve elements from the data dictionary'
            ' are now deprecated. We have upgraded the features of the data'
            ' dictionary. It can now be used directly to retrieve elements'
            " and provide defaults if they don't exist. This warning will"
            ' become an error in the future.',
            DeprecationWarning,
        )
        return data.connectivity

    @staticmethod
    def get_target(
        _: Any,
        data: PassDataPGS,
    ) -> UnitaryMatrix | StateVector | StateSystem:
        """
        Retrieve the target from the data dictionary.

        (Deprecated)
        """
        warnings.warn(
            'BasePass calls to retrieve elements from the data dictionary'
            ' are now deprecated. We have upgraded the features of the data'
            ' dictionary. It can now be used directly to retrieve elements'
            " and provide defaults if they don't exist. This warning will"
            ' become an error in the future.',
            DeprecationWarning,
        )
        return data.target

    @staticmethod
    def in_parallel(data: dict[str, Any]) -> bool:
        """
        Return true if pass is being executed in a parallel.

        (Deprecated)
        """
        warnings.warn(
            'BasePass calls to `in_parallel` are deprecated and will always'
            ' return True. This warning will become an error in the future.',
            DeprecationWarning,
        )
        return True

    @staticmethod
    def execute(*args: Any, **kwargs: Any) -> Any:
        """
        Map a function over iterable arguments in parallel.

        (Defunct)
        """
        raise RuntimeError(
            'Since Dask was removed, the execute function became defunct.'
            ' We have switched to using `get_runtime().map(...)`. You can'
            ' import `get_runtime` from `bqskit.runtime`, and alongside a'
            ' an `await` keyword, should be a  drop-in replacement for the'
            ' execute function. See the following link for more info: '
            'https://bqskit.readthedocs.io/en/latest/source/runtime.html'
            'In a future version, this error will become an AttributeError.',
        )


async def _sub_do_work(
    workflow: WorkflowPGS,
    circuit: Circuit,
    data: PassDataPGS,
) -> tuple[Circuit, PassDataPGS]:
    """Execute a sequence of passes on circuit."""
    if 'calculate_error_bound' in data and data['calculate_error_bound']:
        old_utry = circuit.get_unitary()

    await workflow.run(circuit, data)

    if 'calculate_error_bound' in data and data['calculate_error_bound']:
        new_utry = circuit.get_unitary()
        data.error = new_utry.get_distance_from(old_utry)

    return circuit, data