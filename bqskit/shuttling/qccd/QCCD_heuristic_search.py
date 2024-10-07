from __future__ import annotations

from bqskit import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost import CostFunctionGenerator
from bqskit.ir.opt.cost import HilbertSchmidtCostGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.qis import UnitaryMatrix, StateVector, StateSystem
from bqskit.shuttling.qccd.evaluate_circuit import evaluate_small_circuit
from bqskit.utils.typing import is_real_number


class QCCDHeuristicFunction(HeuristicFunction):
    """
    Heuristic function
    """

    def __init__(
            self,
            heuristic_factor: float = 10.0,
            cost_factor: float = 1.0,
            machine_model: MachineModel = None,
            ion_assignment: dict = None,
            cost_gen: CostFunctionGenerator = HilbertSchmidtCostGenerator(),
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

        self.heuristic_factor = heuristic_factor
        self.cost_factor = cost_factor
        self.cost_gen = cost_gen
        self.QCCD_model = machine_model
        self.ion_assignement = ion_assignment

    def get_value(
            self,
            circuit: Circuit,
            target: UnitaryMatrix | StateVector | StateSystem,
    ) -> float:
        cost = evaluate_small_circuit(circuit=circuit,
                                      QCCD_model=self.QCCD_model,
                                      starting_ion_assignment=self.ion_assignement)
        heuristic = self.cost_gen.calc_cost(circuit, target)
        return self.heuristic_factor * heuristic + self.cost_factor * cost
