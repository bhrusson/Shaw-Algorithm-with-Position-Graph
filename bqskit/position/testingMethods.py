import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw
import matplotlib.pyplot as plt
from enum import Enum
from typing import Tuple, List, Sequence, Mapping, Dict, Callable, Optional
from enum import Enum, IntFlag
from dataclasses import dataclass
from bqskit.position.graph import *
from .graph import *
def testLabels() -> None:
        weights_list = [0,1,2,3,4,5,6,7]
        for i in range(8):
            edgeLabel_i = EdgeLabel(i,{i:i + (i*0.27)})
            print("i:" + str(i) +" is_none() :")
            print(edgeLabel_i.is_none())

            for j in range(8):
                try:
                    print("i:" + str(i) +" has_capability:" + str(j) +"  :" + str(edgeLabel_i.has_capability(EdgeCapability(j))))
                    print()

                    print("i:" + str(i) +" get_weight:" + str(j) +"  :" + str(edgeLabel_i.get_weight(EdgeCapability(j))))
                    print()

                except Exception as e:
                    print(f"An error occurred: {e}")
                finally:
                    print("continuing. . .")
            print("Thank You")
        print("You're Welcome!")
   
def make_two_node_move_graph() -> PositionGraph:
        pos_labels = [
            PositionLabel(PositionCapability.STARTING, {PositionCapability.STARTING: 0.0}),
            PositionLabel(PositionCapability.EXECUTE, {PositionCapability.EXECUTE: 0.0}),
        ]

        edge_labels = {
            (0, 1): EdgeLabel(EdgeCapability.MOVE, {EdgeCapability.MOVE: 1.0}),
            (1, 0): EdgeLabel(EdgeCapability.MOVE, {EdgeCapability.MOVE: 1.0}),
        }

        return PositionGraph(radices=[2], pos_labels=pos_labels, edge_labels=edge_labels)

def make_three_node_execute_cluster() -> PositionGraph:
    pos_labels = [
        PositionLabel(PositionCapability.EXECUTE, {PositionCapability.EXECUTE: 0.1}),
        PositionLabel(PositionCapability.EXECUTE, {PositionCapability.EXECUTE: 0.2}),
        PositionLabel(PositionCapability.EXECUTE, {PositionCapability.EXECUTE: 0.3}),
    ]
    edge_labels = {}
    # fully connected directed subgraph (both directions)
    for i in range(3):
        for j in range(3):
            if i != j:
                edge_labels[(i, j)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.EXECUTE: 0.5}
                )

    return PositionGraph(radices=[2] * len(pos_labels), pos_labels=pos_labels, edge_labels=edge_labels)

def make_16_node_sc_graph() -> PositionGraph:
    pos_labels = []
    edge_labels = {}
    for i in range(16):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 0.1}))
        for j in range(16):
            if i != j:
                edge_labels[i,j] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.EXECUTE: 0.5}
                )
    return PositionGraph(radices=[2] * len(pos_labels), pos_labels=pos_labels, edge_labels=edge_labels)

def make_sc_2_cluster_graph() -> PositionGraph:
    pos_labels = []
    edge_labels = {}
    for i in range(4):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 0.1}))
        for j in range(4):
            if i != j:
                edge_labels[i,j] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.EXECUTE: 0.5}
                )
    pos_labels.append(PositionLabel(PositionCapability.NONE,{}))
    pos_labels.append(PositionLabel(PositionCapability.NONE,{}))   
    pos_labels.append(PositionLabel(PositionCapability.NONE,{}))

    edge_labels[3,4] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[4,3] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[4,5] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[5,4] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[5,6] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[6,5] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[6,7] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[7,6] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})

    for i in range(4):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 0.1}))
        for j in range(4):
            if i != j:
                edge_labels[i+7,j+7] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.EXECUTE: 0.5}
                )

    return PositionGraph(radices=[2] * len(pos_labels), pos_labels=pos_labels, edge_labels=edge_labels)

def testGraphs() -> None:
    g = make_two_node_move_graph()
    print(g.position_has_capability(0, PositionCapability.STARTING))  # True
    print(g.get_valid_starting_positions())                           # [0]
    print(list(g.graph.edge_list()))                                   # [(0,1), (1,0)]

    g = make_three_node_execute_cluster()
    print(g.executable_clusters())  # Expect [[0, 1, 2]]

    g = make_16_node_sc_graph()
    print(g.executable_clusters())

    g = make_sc_2_cluster_graph()
    print(str(g))
    print(g.executable_clusters())
    print(g.connected_to_executable_clusters())
    # MOVE projected graph
    move_g = g.get_projected_graph(EdgeCapability.MOVE)
    mpl_draw(
        move_g,
        with_labels=True,
        node_color='lightblue',
    )
    plt.title("MOVE projected graph")
    plt.show()

    # EXECUTE projected graph
    exec_g = g.get_projected_graph(EdgeCapability.EXECUTE)
    mpl_draw(
        exec_g,
        with_labels=True,
        node_color='orange',
    )
    plt.title("EXECUTE projected graph")
    plt.show()

    g = make_sc_2_cluster_graph2()
    print(g.executable_clusters())

def make_sc_2_cluster_graph2() -> PositionGraph:
    pos_labels = []
    edge_labels = {}

    # --- Cluster 1 (0–3)
    for i in range(4):
        pos_labels.append(PositionLabel(
            PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING,
            {PositionCapability.EXECUTE: 0.1}
        ))
        for j in range(4):
            if i != j:
                edge_labels[i, j] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.EXECUTE: 0.5}
                )

    # --- Bridge nodes (4–6)
    for _ in range(3):
        pos_labels.append(PositionLabel(PositionCapability.NONE, {}))

    for i in range(4, 7):
        if i < 6:
            edge_labels[i, i+1] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP,
                                            {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
            edge_labels[i+1, i] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP,
                                            {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})

    # Connect clusters to bridge
    edge_labels[3, 4] = EdgeLabel(EdgeCapability.MOVE, {EdgeCapability.MOVE: 1.0})
    edge_labels[4, 3] = EdgeLabel(EdgeCapability.MOVE, {EdgeCapability.MOVE: 1.0})
    edge_labels[6, 7] = EdgeLabel(EdgeCapability.MOVE, {EdgeCapability.MOVE: 1.0})
    edge_labels[7, 6] = EdgeLabel(EdgeCapability.MOVE, {EdgeCapability.MOVE: 1.0})

    # --- Cluster 2 (7–10)
    for i in range(4):
        pos_labels.append(PositionLabel(
            PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING,
            {PositionCapability.EXECUTE: 0.1}
        ))
        for j in range(4):
            if i != j:
                edge_labels[i+7, j+7] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.EXECUTE: 0.5}
                )

    return PositionGraph(radices=[2]*len(pos_labels), pos_labels=pos_labels, edge_labels=edge_labels)


testLabels()
testGraphs()
    


