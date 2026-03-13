import numpy as np
import bqskit
import pkgutil
import rustworkx as rx
from rustworkx.visualization import mpl_draw
import matplotlib.pyplot as plt
from enum import Enum
from typing import Tuple, List, Sequence, Mapping, Dict, Callable, Optional
from enum import Enum, IntFlag
from dataclasses import dataclass
from bqskit.compiler.gateset import GateSetLike
from bqskit.compiler.gateset import GateSet
from bqskit_local.position.graph import *
from .graph import *

def testLabels() -> None:
        weights_list = [0,1,2,3,4,5,6,7]
        for i in range(8):
            edgeLabel_i = EdgeLabel(i,{i:i + (i*0.27)})
            print("i:" + str(i) +" is_none() :")
            #print(edgeLabel_i.is_none())

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


        print([name for _, name, _ in pkgutil.iter_modules(bqskit.__path__)])



def testGraphs() -> None:
    """
    g = make_32_node_sparse_graph()
    
    pos_graph_plot_stats(g)
    print(str(g._executable_clusters))

    g = make_2_cluster_graph()
    pos_graph_plot_stats(g)
    print(str(g._executable_clusters))

   g = make_16_node_sc_graph()
    pos_graph_plot_stats(g)

    g = make_2_cluster_graph()
    pos_graph_plot_stats(g)

    g = make_2_cluster_graph_notFC()
    pos_graph_plot_stats(g)

    g = make_connected_loop()
    #pos_graph_plot_stats(g)

    g = make_2_connected_loop()
    pos_graph_plot_stats(g)

    g = make_2_connected_loop_limit_move()
    pos_graph_plot_stats(g)

    g = make_connected_pairs()
    pos_graph_plot_stats(g)"""
    



def pos_graph_plot_stats(g:PositionGraph) -> None:
    #print(str(g))
    print(g.executable_clusters())
    print(g.executable_clusters2())     
    print(g.executable_clusters3())
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
    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)


def make_10_node_ring_position_graph() -> PositionGraph:
    num_nodes = 10
    pos_labels = []
    edge_labels = {}

    for _ in range(num_nodes):
        pos_labels.append(
            PositionLabel(
                PositionCapability.EXECUTE
                | PositionCapability.MEASURE
                | PositionCapability.STARTING,
                {PositionCapability.EXECUTE: 0.1},
            )
        )

    for i in range(num_nodes):
        j = (i + 1) % num_nodes

        edge_labels[(i, j)] = EdgeLabel(
            EdgeCapability.MOVE | EdgeCapability.EXECUTE,
            {EdgeCapability.MOVE: 1.0, EdgeCapability.EXECUTE: 0.5},
        )
        edge_labels[(j, i)] = EdgeLabel(
            EdgeCapability.MOVE | EdgeCapability.EXECUTE,
            {EdgeCapability.MOVE: 1.0, EdgeCapability.EXECUTE: 0.5},
        )

    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)
    
def make_32_node_sparse_graph() -> PositionGraph:
    pos_labels = []
    edge_labels = {}

    num_nodes = 12
    for i in range(num_nodes):
        pos_labels.append(PositionLabel(
            PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING,
            {PositionCapability.EXECUTE: 0.1}
        ))

    # Connect neighbors in a sparse linear fashion
    
    for i in range(num_nodes - 1):
        edge_labels[i, i+1] = EdgeLabel(
            EdgeCapability.SWAP | EdgeCapability.EXECUTE,
            {EdgeCapability.SWAP: 1.0, EdgeCapability.EXECUTE: 0.5}
        )
        edge_labels[i+1, i] = EdgeLabel(
            EdgeCapability.SWAP | EdgeCapability.EXECUTE,
            {EdgeCapability.SWAP: 1.0, EdgeCapability.EXECUTE: 0.5}
    )
    edge_labels[0,num_nodes-1] = EdgeLabel(
        EdgeCapability.SWAP | EdgeCapability.EXECUTE,
        {EdgeCapability.SWAP: 1.0, EdgeCapability.EXECUTE: 0.5}
    )
    edge_labels[num_nodes-1,0] = EdgeLabel(
        EdgeCapability.SWAP | EdgeCapability.EXECUTE,
        {EdgeCapability.SWAP: 1.0, EdgeCapability.EXECUTE: 0.5}
    )
    
    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)

def make_line_graph(size) -> PositionGraph:
    pos_labels = []
    edge_labels = {}

    for i in range(size):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 1.0}))

    for i in range(size-1):
        edge_labels[(i,i+1)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.00, EdgeCapability.EXECUTE: 1.00})
        edge_labels[(i+1,i)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.00, EdgeCapability.EXECUTE: 1.00})
                    
    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)


def make_all_connected(size) -> PositionGraph:
    pos_labels = []
    edge_labels = {}

    for i in range(size):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 1.0}))

    for i in range(size-1):
        edge_labels[(i,i+1)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.00, EdgeCapability.EXECUTE: 1.00})
        edge_labels[(i+1,i)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.00, EdgeCapability.EXECUTE: 1.00})
    for i in range(size-2):
        edge_labels[(i,i+2)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.00, EdgeCapability.EXECUTE: 1.00})
        edge_labels[(i+2,i)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.00, EdgeCapability.EXECUTE: 1.00})
    for i in range(size-3):
        edge_labels[(i,i+3)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.00, EdgeCapability.EXECUTE: 1.00})
        edge_labels[(i+3,i)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.00, EdgeCapability.EXECUTE: 1.00})
                    
    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)


def make_2_cluster_graph() -> PositionGraph:
    num_nodes_per_cluster = 13
    pos_labels = []
    edge_labels = {}
    for i in range(num_nodes_per_cluster):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 0.1}))
        for j in range(num_nodes_per_cluster):
            if i != j:
                edge_labels[i,j] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.EXECUTE: 0.5}
                )
    pos_labels.append(PositionLabel(PositionCapability.NONE,{}))
    pos_labels.append(PositionLabel(PositionCapability.NONE,{}))   
    pos_labels.append(PositionLabel(PositionCapability.NONE,{}))

    
    edge_labels[num_nodes_per_cluster - 1, num_nodes_per_cluster] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[num_nodes_per_cluster, num_nodes_per_cluster - 1] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[num_nodes_per_cluster , num_nodes_per_cluster + 1] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[num_nodes_per_cluster + 1 ,num_nodes_per_cluster] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[num_nodes_per_cluster + 1, num_nodes_per_cluster + 2] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[num_nodes_per_cluster + 2, num_nodes_per_cluster + 1] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[num_nodes_per_cluster + 2, num_nodes_per_cluster + 3] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    edge_labels[num_nodes_per_cluster + 3, num_nodes_per_cluster + 2] = EdgeLabel(EdgeCapability.MOVE | EdgeCapability.SWAP, {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 1.5})
    

    for i in range(num_nodes_per_cluster):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 0.1}))
        for j in range(num_nodes_per_cluster):
            if i != j:
                edge_labels[i+num_nodes_per_cluster + 3,j+num_nodes_per_cluster + 3] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.EXECUTE: 0.5}
                )

    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)

def make_connected_loop() -> PositionGraph:
    pos_labels = []
    edge_labels = {}
    size = 7

    for i in range(size):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 0.1}))
        edge_labels[i,(i+1)%size] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75, EdgeCapability.EXECUTE: 0.5}
                )
        edge_labels[i,((i-1)%size)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75, EdgeCapability.EXECUTE: 0.5}
                )
    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)  


def make_2_connected_loop() -> PositionGraph:
    pos_labels = []
    edge_labels = {}
    size = 5

    for i in range(size):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 0.1}))
        edge_labels[i,(i+1)%size] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75, EdgeCapability.EXECUTE: 0.5}
                )
        edge_labels[i,((i-1)%size)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75, EdgeCapability.EXECUTE: 0.5}
                )
        print(i)
        print((i+1)%size)
        print(((i-1)%size))
        
    pos_labels.append(PositionLabel(PositionCapability.NONE, {PositionCapability.EXECUTE: 0.1}))

    for i in range(size):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 0.1}))
        edge_labels[i+ 1 + size,(((i+1)%size) + 1 + size)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75, EdgeCapability.EXECUTE: 0.5}
                )
        edge_labels[i+ 1 + size,((i-1)%size + 1 + size)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75, EdgeCapability.EXECUTE: 0.5}
                )
        print(i + 1 + size)
        print(((i+1)%size) + 1 + size)
        print(((i-1)%size + 1 + size))
    edge_labels[size,size+1] = EdgeLabel(
        EdgeCapability.MOVE | EdgeCapability.SWAP,
        {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75}
    )
    edge_labels[size,size-1] = EdgeLabel(
        EdgeCapability.MOVE | EdgeCapability.SWAP,
        {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75}
    )
    edge_labels[size+1,size] = EdgeLabel(
        EdgeCapability.MOVE | EdgeCapability.SWAP,
        {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75}
    )
    edge_labels[size-1,size] = EdgeLabel(
        EdgeCapability.MOVE | EdgeCapability.SWAP,
        {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75}
    )

    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)  


def make_connected_pairs() -> PositionGraph:
    pos_labels = []
    edge_labels = {}
    size = 12

    for i in range(size):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 0.1}))

        if (i%2):
            edge_labels[i,(i+1)%size] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75, EdgeCapability.EXECUTE: 0.5}
                )
            edge_labels[(i+1)%size,i] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75, EdgeCapability.EXECUTE: 0.5}
                )



def make_2_connected_loop_limit_move() -> PositionGraph:
    pos_labels = []
    edge_labels = {}
    size = 5

    for i in range(size):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 0.1}))
        edge_labels[i,(i+1)%size] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75, EdgeCapability.EXECUTE: 0.5}
                )
        edge_labels[i,((i-1)%size)] = EdgeLabel(
                     EdgeCapability.EXECUTE, 
                    {EdgeCapability.EXECUTE: 0.5}
                )
        print(i)
        print((i+1)%size)
        print(((i-1)%size))
        
    pos_labels.append(PositionLabel(PositionCapability.NONE, {PositionCapability.EXECUTE: 0.1}))

    for i in range(size):
        pos_labels.append(PositionLabel(PositionCapability.EXECUTE | PositionCapability.MEASURE | PositionCapability.STARTING, {PositionCapability.EXECUTE: 0.1}))
        edge_labels[i+ 1 + size,(((i+1)%size) + 1 + size)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75, EdgeCapability.EXECUTE: 0.5}
                )
        edge_labels[i+ 1 + size,((i-1)%size + 1 + size)] = EdgeLabel(
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE | EdgeCapability.SWAP,
                    {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75, EdgeCapability.EXECUTE: 0.5}
                )
        print(i + 1 + size)
        print(((i+1)%size) + 1 + size)
        print(((i-1)%size + 1 + size))
    edge_labels[size,size+1] = EdgeLabel(
        EdgeCapability.MOVE | EdgeCapability.SWAP,
        {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75}
    )
    edge_labels[size,size-1] = EdgeLabel(
        EdgeCapability.MOVE | EdgeCapability.SWAP,
        {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75}
    )
    edge_labels[size+1,size] = EdgeLabel(
        EdgeCapability.MOVE | EdgeCapability.SWAP,
        {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75}
    )
    edge_labels[size-1,size] = EdgeLabel(
        EdgeCapability.MOVE | EdgeCapability.SWAP,
        {EdgeCapability.MOVE: 1.0, EdgeCapability.SWAP: 0.75}
    )

    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)  



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
                    EdgeCapability.MOVE | EdgeCapability.EXECUTE ,
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

    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)


    

#testLabels()
testGraphs()

