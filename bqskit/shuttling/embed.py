"""This module implements the EmbedAllPermutationsPass pass."""
from __future__ import annotations

import copy
import itertools as it
import numpy as np
import logging
from typing import Callable
from .gatezone_selection import GateZoneSelectionPass
from .layergen import ShuttlingLayerGenerator
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes.mapping.topology import SubtopologySelectionPass
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.permutation import PermutationMatrix
from bqskit.runtime import get_runtime
from pytket.phir.qtm_machine import QtmMachine

_logger = logging.getLogger(__name__)


def multi_qudit_op_count(circuit: Circuit) -> float:
    """Counts the number of multi-qudit operations in a circuit."""
    x = sum(c for g, c in circuit.gate_counts.items() if g.num_qudits >= 2)
    return float(x)


def from_cg_to_zone(cg: set):
    # {int}: left
    # {float}: right
    if cg == CouplingGraph({(0, 2), (1, 2)}):
        return [{2.1}, {2.0}, {0, 1}]
    elif cg == CouplingGraph({(0, 1), (0, 2)}):
        return [{0.2}, {0.1}, {1, 2}]
    elif cg == CouplingGraph({(0, 1), (1, 2)}):
        return [{1.2}, {1.0}, {0, 2}]
    else:
        raise ValueError("Invalid cg")


class ShuttlingEmbedAllPermutationsPass(BasePass):
    """Embed permutation aware synthesis results into a flow for future use."""

    def __init__(
            self,
            input_perm: bool = False,
            output_perm: bool = True,
            vary_topology: bool = True,
            vary_gatezone: bool = True,
            inner_synthesis: SynthesisPass = LEAPSynthesisPass(),
            scoring_fn: Callable[[Circuit], float] = multi_qudit_op_count,
            qtm_machine: QtmMachine = QtmMachine.H1_1,
    ) -> None:
        """
        Construct a EmbedAllPermutationsPass.

        Args:
            input_perm (bool): If true, vary the input permutation
                during synthesis. (Default: False)

            output_perm (bool): If true, vary the output permutation
                during synthesis. (Default: True)

            vary_topology (bool): If true, vary the desired coupling graph
                during synthesis. (Default: True)

            inner_synthesis (SynthesisPass): The synthesis algorithm used
                on all permutations. (Default: :class:`LEAPSynthesisPass`)

            scoring_fn (Callable[[Circuit], float]): The scoring function
                used when comparing the original circuit to the synthesized
                circuit with the same configuration. The smallest score wins.
                (Default: :func:`multi_qudit_op_count`)
        """

        if not isinstance(inner_synthesis, SynthesisPass):
            bad_type = type(inner_synthesis)
            raise TypeError(f'Expected SynthesisPass object, got {bad_type}.')

        if not callable(scoring_fn):
            bad_type = type(scoring_fn)
            m = f'Expected a function from circuits to scores, got {bad_type}.'
            raise TypeError(m)

        self.input_perm = input_perm
        self.output_perm = output_perm
        self.vary_topology = vary_topology
        self.vary_gatezone = vary_gatezone
        self.inner_synthesis = inner_synthesis
        self.scoring_fn = scoring_fn
        self.qtm_machine = qtm_machine

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        utry = data.target

        if not all(r == utry.radixes[0] for r in utry.radixes):
            raise NotImplementedError(
                'PermutationAwareSynthesisPass only supports unitaries '
                'with the same radix on all qudits currently.',
            )

        # Calculate all permuted targets
        width = utry.num_qudits
        perms = list(it.permutations(range(width)))
        no_perm = [tuple(range(width))]
        Pis = [
            PermutationMatrix.from_qudit_location(width, utry.radixes[0], p)
            for p in perms
        ]
        Pos = [
            PermutationMatrix.from_qudit_location(width, utry.radixes[0], p)
            for p in perms
        ]

        if self.input_perm and self.output_perm:
            permsbyperms = list(it.product(perms, perms))
            targets = [Po.T @ utry @ Pi for Pi, Po in it.product(Pis, Pos)]

        elif self.input_perm:
            permsbyperms = list(it.product(perms, no_perm))
            targets = [utry @ Pi for Pi in Pis]

        elif self.output_perm:
            permsbyperms = list(it.product(no_perm, perms))
            targets = [Po.T @ utry for Po in Pos]

        else:
            _logger.warning('No permutation is being used in PAS.')
            permsbyperms = list(it.product(no_perm, no_perm))
            targets = [utry]

        # Calculate all target coupling graphs
        if self.vary_topology and width != 1:
            if SubtopologySelectionPass.key not in data:
                raise RuntimeError(
                    'Cannot find subtopologies, try running a'
                    ' SubtopologySelectionPass first.',
                )

            if width not in data[SubtopologySelectionPass.key]:
                raise RuntimeError(
                    'Subtopology information for block size'
                    f' {width} is not available.',
                )
            print("All topologies: ", data[SubtopologySelectionPass.key])
            graphs = data[SubtopologySelectionPass.key][width]
        else:
            graphs = [CouplingGraph.all_to_all(width)]
        print("Graphs: ", graphs)

        # Calculate the possible gate zones
        '''
        Consider the case when there exits gate zone place in block
        Assume that no tq_zone is next to each other
        '''
        if self.vary_gatezone and width != 1:
            if GateZoneSelectionPass.key not in data:
                raise RuntimeError(
                    'Cannot find gatezone, try running a'
                    ' GateZoneSelectionPass first.',
                )
            for possible_gatezone_amount in range(1, ((width + 1) // 2) + 1):
                if possible_gatezone_amount not in data[GateZoneSelectionPass.key]:
                    raise RuntimeError(
                        'Possible gate zone information for block size'
                        f' {width} is not available.',
                    )
            gate_zones = data[GateZoneSelectionPass.key]
        else:
            gate_zones = set(i for i in range(0, width, 2))
        print("Gate zones: ", gate_zones)

        # Manually restrict the gate zone and coupling graph to H1 machine configurations
        if self.qtm_machine == QtmMachine.H1_1:
            graphs = [CouplingGraph({(0, 2), (1, 2)}),
                      CouplingGraph({(0, 1), (0, 2)}),
                      CouplingGraph({(0, 1), (1, 2)})]

        extended_gate_zones = []
        # Distribute subgraph connectivity and gate zone to data
        datas = []
        for graph in graphs:
            for zone in from_cg_to_zone(graph):
                model = MachineModel(
                    circuit.num_qudits, graph,
                    data.gate_set, data.model.radixes,
                )
                target_data = copy.deepcopy(data)
                target_data.model = model
                target_data[ShuttlingLayerGenerator.key] = zone
                extended_gate_zones.append(zone)
                print(f"Connectivity: {target_data.connectivity} with zone {zone}")
                datas.append(target_data)

        print("Extended gate zone :", extended_gate_zones)
        # Create parallel arrays for map
        print("Amount of target: ", len(targets))
        print("Amount of data: ", len(datas))
        extended_targets = []
        extended_datas = []
        for t, d in it.product(targets, datas):
            extended_targets.append(t)
            extended_datas.append(d)
        print("Amount of target: ", len(extended_targets))
        print("Amount of data: ", len(extended_datas))

        # Synthesize all permuted targets
        circuits: list[Circuit] = await get_runtime().map(
            self.inner_synthesis.synthesize,
            extended_targets,  # extend target
            extended_datas,  # modify
        )
        # Store results
        zone_perm_data: dict[tuple[int, ...], dict[
            CouplingGraph,
            dict[tuple[tuple[int, ...], tuple[int, ...]], Circuit]],
        ] = {}
        for i, c in enumerate(circuits):
            # print(f"Perm data at {i} in the begining: {zone_perm_data}")
            print(f"Index: {i}")
            zone = extended_gate_zones[i % len(extended_gate_zones)]
            graph_interval = int(len(extended_gate_zones) / len(graphs))
            graph = graphs[(i // graph_interval) % len(graphs)]
            perm = permsbyperms[i // (len(graphs) * graph_interval)]
            print("Current zone:", zone)
            print("Current graph: ", graph)
            print("Permutation: ", perm)
            print("Circuit connectivity:", c.coupling_graph)
            print("Circuit QASM:", c.to('qasm'))
            zone = tuple(zone)
            if zone not in zone_perm_data:
                zone_perm_data[zone] = {}
            if graph not in zone_perm_data[zone]:
                zone_perm_data[zone][graph] = {}

            if perm in zone_perm_data[zone][graph]:
                # Update if it is better than whats already there
                s1 = self.scoring_fn(zone_perm_data[zone][graph][perm])
                s2 = self.scoring_fn(c)
                if s2 < s1:
                    zone_perm_data[zone][graph][perm] = c
            else:
                zone_perm_data[zone][graph][perm] = c

            # print(f"Perm data at {i} after update: {zone_perm_data}")

            # Calculate number of multi-qudit gates
            num_mq_gates = 0
            for gate, count in c.gate_counts.items():
                if gate.num_qudits >= 2:
                    num_mq_gates += count

            # Generate the extra circuits through universal permutations
            print("Universal permutation")
            all_perms = list(it.permutations(range(width)))
            for univ_perm in all_perms[1:]:
                renumber_c = c.copy()
                renumber_c.renumber_qudits(univ_perm)
                new_pi = tuple(univ_perm[i] for i in perm[0])
                new_pf = tuple(univ_perm[i] for i in perm[1])
                #print("Input permutation: ", new_pi)
                new_graph = renumber_c.coupling_graph
                #print("Old zone: ", zone)
                new_zone = list()  # TODO: return gate_zone after rotate by universal permutations
                if len(zone) > 1:
                    for z in zone:
                        new_zone.append(new_pi[z])
                else:
                    z = list(zone)[0]
                    z1 = int(z)
                    z2 = int(np.round((z - int(z)) * 10))
                    new_zone.append(float(new_pi[z1] + 0.1 * new_pi[z2]))

                new_zone = tuple(new_zone)
                # print("New zone: ", new_zone)
                if new_zone not in zone_perm_data:
                    zone_perm_data[new_zone] = {}

                if new_graph not in zone_perm_data[new_zone]:
                    zone_perm_data[new_zone][new_graph] = {}

                new_perm = (new_pi, new_pf)
                if new_perm not in zone_perm_data[new_zone][new_graph]:
                    zone_perm_data[new_zone][new_graph][new_perm] = renumber_c
                else:
                    s1 = self.scoring_fn(zone_perm_data[new_zone][new_graph][new_perm])
                    s2 = self.scoring_fn(renumber_c)
                    if s2 < s1:
                        zone_perm_data[new_zone][new_graph][new_perm] = renumber_c
            # print(f"Perm data at {i} after generating the extra circuits through universal permutations: {zone_perm_data}")

        # Override no perm result if original is better and compatible
        if circuit.gate_set.issubset(data.model.gate_set):
            for univ_perm in it.permutations(range(width)):
                # Permute original circuit and override worse results
                uperm = (univ_perm, univ_perm)
                renumber_c = circuit.copy()
                renumber_c.renumber_qudits(univ_perm)
                new_graph = renumber_c.coupling_graph
                new_zone = data[ShuttlingLayerGenerator.key]  # TODO: return gate_zone given a circuit after renumbering
                new_score = self.scoring_fn(renumber_c)
                for zone, zone_data in zone_perm_data.items():
                    for graph, graph_data in zone_data.items():
                        if all(z in zone for z in new_zone):
                            if all(e in graph for e in new_graph):
                                if uperm not in graph_data:
                                    graph_data[uperm] = renumber_c
                                else:
                                    if new_score < self.scoring_fn(graph_data[uperm]):
                                        graph_data[uperm] = renumber_c

        # Record permutation data in the pass data
        data['permutation_data'] = zone_perm_data
