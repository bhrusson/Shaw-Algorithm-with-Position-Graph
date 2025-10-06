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

def testGraphs() -> None:
    g = make_16_node_sc_graph()
    print(g.executable_clusters())
    print(g.executable_clusters2())
    g = make_2_cluster_graph()
    #print(str(g))
    print(g.executable_clusters())
    print(g.executable_clusters2())
    print(g.connected_to_executable_clusters())
    # MOVE projected graph
    move_g = g.get_projected_graph(EdgeCapability.MOVE)
    mpl_draw(
        move_g,
        with_labels=True,
        node_color='lightblue',
    )
    plt.title("MOVE projected graph")
    #plt.show()

    # EXECUTE projected graph
    exec_g = g.get_projected_graph(EdgeCapability.EXECUTE)
    mpl_draw(
        exec_g,
        with_labels=True,
        node_color='orange',
    )
    plt.title("EXECUTE projected graph")
    #plt.show()

    g = make_2_cluster_graph_notFC()
    print(g.executable_clusters())
    print(g.executable_clusters2())
    print(g.connected_to_executable_clusters())

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

def make_2_cluster_graph() -> PositionGraph:
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



def make_2_cluster_graph_notFC() -> PositionGraph:
    pos_labels = []
    edge_labels = {}
    pos_labels.append(PositionLabel(PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 0.1}))

    for i in range(1 , 4):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 0.1}))
        edge_labels[0,i] = EdgeLabel(
                    EdgeCapability.MOVE,
                    {EdgeCapability.MOVE: 1.0}
                )
        edge_labels[i,0] = EdgeLabel(
                    EdgeCapability.MOVE,
                    {EdgeCapability.MOVE: 1.0}
                )        
        for j in range(1,4):
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


    

testLabels()
testGraphs()

