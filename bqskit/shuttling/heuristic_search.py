from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost import CostFunctionGenerator
from bqskit.ir.opt.cost import HilbertSchmidtCostGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.qis import UnitaryMatrix, StateVector, StateSystem
from pytket.phir.qtm_machine import QtmMachine
from bqskit.shuttling.util import get_gate_time, get_duration_from_circ
from bqskit.utils.typing import is_real_number



class HeuristicSearch(HeuristicFunction):
    """
    Heuristic function
    """

    def __init__(
            self,
            heuristic_factor: float = 10.0,
            cost_factor: float = 1.0,
            cost_gen: CostFunctionGenerator = HilbertSchmidtCostGenerator(),
            qtm_machine: QtmMachine = QtmMachine.H1_1,
    ) -> None:
        """
        Construct a AStarHeuristic Function.

        Args:
            heuristic_factor (float): Scale the heuristic component by
                this value.

            cost_factor (float): Scale the cost component by this value.

            cost_gen (CostFunctionGenerator): This is used to generate
                cost functions used during evaluations.
        """
        if not is_real_number(heuristic_factor):
            raise TypeError(
                'Expected float for heuristic_factor, got %s.'
                % type(heuristic_factor),
            )

        if not is_real_number(cost_factor):
            raise TypeError(
                'Expected float for cost_factor, got %s.'
                % type(cost_factor),
            )

        if not isinstance(cost_gen, CostFunctionGenerator):
            raise TypeError(
                'Expected CostFunctionGenerator for cost_gen, got %s.'
                % type(cost_gen),
            )

        if not isinstance(qtm_machine, QtmMachine):
            raise TypeError(
                'Expected QtmMachine type , got %s.'
                % type(qtm_machine),
            )

        self.heuristic_factor = heuristic_factor
        self.cost_factor = cost_factor
        self.cost_gen = cost_gen
        self.qtm_machine = qtm_machine

    def get_value(
            self,
            circuit: Circuit,
            target: UnitaryMatrix | StateVector | StateSystem,
    ) -> float:
        cost = get_duration_from_circ(circuit, self.qtm_machine)
        # cost = 0
        # for op in circuit.gate_set:
        #     if op == SwapGate():
        #         cost += circuit.count(op)*0.9
        #     elif op.num_qudits == 2:
        #         cost += circuit.count(op)*0.04
        #     elif op.num_qudits == 1:
        #         cost += circuit.count(op)*0.03
        heuristic = self.cost_gen.calc_cost(circuit, target)
        return self.heuristic_factor * heuristic + self.cost_factor * cost
