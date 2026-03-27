from __future__ import annotations

import copy
import itertools as it
from typing import Any, Iterator, MutableMapping, Sequence

from bqskit.compiler.machine import MachineModel
from bqskit.compiler.gateset import GateSet
from bqskit.ir.circuit import Circuit
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer, is_real_number, is_sequence

from bqskit_local.position.graph import PositionGraph
from bqskit_local.position.state import PositionGraphState


class PassDataPGS(MutableMapping[str, Any]):
    """
    A PassData variant for PositionGraph / PositionGraphState workflows.
    """

    _reserved_keys = [
        'target',
        'model',
        'placement',
        'error',
        'seed',
        'machine_model',
        'initial_mapping',
        'final_mapping',
        'position_graph',
        'pgs',
    ]

    def __init__(self, circuit: Circuit) -> None:
        self._target: Circuit | StateVector | UnitaryMatrix | StateSystem
        if circuit.num_qudits <= 8:
            try:
                self._target = circuit.get_unitary()
            except RuntimeError:
                self._target = circuit
        else:
            self._target = circuit

        self._error = 0.0
        self._model = MachineModel(circuit.num_qudits)
        self._placement = list(range(circuit.num_qudits))
        self._initial_mapping = list(range(circuit.num_qudits))
        self._final_mapping = list(range(circuit.num_qudits))
        self._position_graph: PositionGraph | None = None
        self._pgs: PositionGraphState | None = None
        self._data: dict[str, Any] = {}
        self._seed: int | None = None

    @property
    def target(self) -> StateVector | UnitaryMatrix | StateSystem:
        if isinstance(self._target, Circuit):
            self._target = self._target.get_unitary()
        return self._target

    @target.setter
    def target(self, _val: StateVector | UnitaryMatrix | StateSystem) -> None:
        if not isinstance(_val, (StateVector, UnitaryMatrix, StateSystem)):
            raise TypeError(
                f'Cannot assign type {type(_val)} to target. '
                'Expected StateVector, StateSystem, or UnitaryMatrix.',
            )
        if len(self.placement) != _val.num_qudits:
            self.placement = list(range(_val.num_qudits))
        self._target = _val

    @property
    def error(self) -> float:
        return self._error

    @error.setter
    def error(self, _val: float) -> None:
        if not is_real_number(_val):
            raise TypeError(
                f'Cannot assign type {type(_val)} to error. '
                'Expected a real number.',
            )
        self._error = float(_val)

    @property
    def model(self) -> MachineModel:
        return self._model

    @model.setter
    def model(self, _val: MachineModel) -> None:
        if not isinstance(_val, MachineModel):
            raise TypeError(
                f'Cannot set model to {type(_val)}. Expected a MachineModel.',
            )
        self._model = _val

    @property
    def gate_set(self) -> GateSet:
        if self._pgs is not None:
            return self._pgs.gateSet
        return self._model.gate_set

    @gate_set.setter
    def gate_set(self, _val: GateSet) -> None:
        if not isinstance(_val, GateSet):
            raise TypeError(
                f'Cannot set gate_set to {type(_val)}. Expected a GateSet.',
            )
        if self._pgs is not None:
            self._pgs._gateSet = _val
        else:
            self._model.gate_set = _val

    @property
    def placement(self) -> list[int]:
        return self._placement

    @placement.setter
    def placement(self, _val: Sequence[int]) -> None:
        if not is_sequence(_val):
            raise TypeError(
                f'Cannot set placement to {type(_val)}. '
                'Expected a sequence of integers.',
            )
        if not all(is_integer(x) for x in _val):
            raise TypeError(
                'Cannot set placement. Expected a sequence of integers.',
            )
        self._placement = list(int(x) for x in _val)

    @property
    def initial_mapping(self) -> list[int]:
        return self._initial_mapping

    @initial_mapping.setter
    def initial_mapping(self, _val: Sequence[int]) -> None:
        if not is_sequence(_val):
            raise TypeError(
                f'Cannot set initial_mapping to {type(_val)}. '
                'Expected a sequence of integers.',
            )
        if not all(is_integer(x) for x in _val):
            raise TypeError(
                'Cannot set initial_mapping. Expected a sequence of integers.',
            )
        self._initial_mapping = list(int(x) for x in _val)

    @property
    def final_mapping(self) -> list[int]:
        return self._final_mapping

    @final_mapping.setter
    def final_mapping(self, _val: Sequence[int]) -> None:
        if not is_sequence(_val):
            raise TypeError(
                f'Cannot set final_mapping to {type(_val)}. '
                'Expected a sequence of integers.',
            )
        if not all(is_integer(x) for x in _val):
            raise TypeError(
                'Cannot set final_mapping. Expected a sequence of integers.',
            )
        self._final_mapping = list(int(x) for x in _val)

    @property
    def seed(self) -> int | None:
        return self._seed

    @seed.setter
    def seed(self, _val: int | None) -> None:
        if _val is not None and not is_integer(_val):
            raise TypeError(
                f'Cannot set seed to {type(_val)}. '
                'Expected an integer or None.',
            )
        self._seed = None if _val is None else int(_val)

    @property
    def position_graph(self) -> PositionGraph:
        if self._position_graph is None:
            raise RuntimeError('No PositionGraph has been set.')
        return self._position_graph

    @position_graph.setter
    def position_graph(self, _val: PositionGraph) -> None:
        if not isinstance(_val, PositionGraph):
            raise TypeError(
                f'Cannot set position_graph to {type(_val)}. '
                'Expected a PositionGraph.',
            )
        self._position_graph = _val

    @property
    def pgs(self) -> PositionGraphState:
        if self._pgs is None:
            raise RuntimeError('No PositionGraphState has been set.')
        return self._pgs

    @pgs.setter
    def pgs(self, _val: PositionGraphState) -> None:
        if not isinstance(_val, PositionGraphState):
            raise TypeError(
                f'Cannot set pgs to {type(_val)}. '
                'Expected a PositionGraphState.',
            )
        self._pgs = _val
        self._position_graph = _val.position_graph

        # Keep standard mapping fields synchronized
        self._placement = list(int(x) for x in _val.logical_to_position)
        self._initial_mapping = list(int(x) for x in _val.logical_to_position)
        self._final_mapping = list(int(x) for x in _val.logical_to_position)

    @property
    def connectivity(self) -> CouplingGraph:
        """
        Return current logical executability as a CouplingGraph.

        For PGS workflows, this is derived from the current PositionGraphState.
        For native workflows, it falls back to model + placement.
        """
        if self._pgs is not None:
            return self._pgs.to_coupling_graph()
        return self.model.coupling_graph.get_subgraph(self.placement)

    def __getitem__(self, _key: str) -> Any:
        if _key in self._reserved_keys:
            if _key == 'machine_model':
                _key = 'model'
            return self.__getattribute__(_key)
        return self._data.__getitem__(_key)

    def __setitem__(self, _key: str, _val: Any) -> None:
        if _key in self._reserved_keys:
            if _key == 'machine_model':
                _key = 'model'
            return self.__setattr__(_key, _val)
        return self._data.__setitem__(_key, _val)

    def __delitem__(self, _key: str) -> None:
        if _key in self._reserved_keys:
            raise RuntimeError(f'Cannot delete {_key} from data.')
        return self._data.__delitem__(_key)

    def __iter__(self) -> Iterator[str]:
        return it.chain(self._reserved_keys.__iter__(), self._data.__iter__())

    def __len__(self) -> int:
        return self._data.__len__() + len(self._reserved_keys)

    def __contains__(self, _o: object) -> bool:
        return self._reserved_keys.__contains__(_o) or self._data.__contains__(_o)

    def update(self, other: Any = (), /, **kwds: Any) -> None:
        if isinstance(other, PassDataPGS):
            for key in other:
                if key == 'target':
                    self._target = other._target
                    continue
                self[key] = other[key]

            for key, value in kwds.items():
                self[key] = value
            return

        super().update(other, **kwds)

    def copy(self) -> PassDataPGS:
        return copy.deepcopy(self)

    def become(self, other: PassDataPGS, deepcopy: bool = False) -> None:
        if deepcopy:
            self._target = copy.deepcopy(other._target)
            self._error = copy.deepcopy(other._error)
            self._model = copy.deepcopy(other._model)
            self._placement = copy.deepcopy(other._placement)
            self._initial_mapping = copy.deepcopy(other._initial_mapping)
            self._final_mapping = copy.deepcopy(other._final_mapping)
            self._position_graph = copy.deepcopy(other._position_graph)
            self._pgs = copy.deepcopy(other._pgs)
            self._data = copy.deepcopy(other._data)
            self._seed = copy.deepcopy(other._seed)
        else:
            self._target = copy.copy(other._target)
            self._error = copy.copy(other._error)
            self._model = copy.copy(other._model)
            self._placement = copy.copy(other._placement)
            self._initial_mapping = copy.copy(other._initial_mapping)
            self._final_mapping = copy.copy(other._final_mapping)
            self._position_graph = copy.copy(other._position_graph)
            self._pgs = copy.copy(other._pgs)
            self._data = copy.copy(other._data)
            self._seed = copy.copy(other._seed)

    def update_error_mul(self, error: float) -> None:
        self.error = (1 - ((1 - self.error) * (1 - error)))