# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Parallel communication pattern analysis
#
# This demo is implemented in {download}`demo_comms_pattern.py`. It
# illustrates how build a graph that represents a parallel communication
# pattern and how to analyse the parallel communication pattern using
# [NetworkX](https://networkx.org/).
#
# The layout of a distributed array across processes (MPI ranks) is
# described in DOLFINx by an IndexMap. It represents the range of
# locally 'owned' array indices and the indices that are ghosted on a
# rank. The IndexMap also holds information on the ranks that the
# calling rank will send data to and ranks that will send data to the
# caller.
#

# +
import itertools as it
import json

from mpi4py import MPI

import matplotlib.pyplot as plt
import networkx as nx

from dolfinx import fem, mesh
from dolfinx.cpp.common import IndexMap

# -


# The following function plots a directed graph, with the edge weights
# labeled. Each node is an MPI rank, and an edge represents a
# communication edge. The edge weights indicate the volume of data
# communicated.


# +
def plot(G: nx.MultiGraph):
    """Plot the communication graph."""
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, alpha=0.75)
    nx.draw_networkx_labels(G, pos, font_size=12)

    # Curve edges to distinguish between in- and out-edges
    connectstyle = [f"arc3,rad={r}" for r in it.accumulate([0.25] * 4)]

    # Color edges by local (shared memory) or remote (remote memory)
    # communication
    width = 0.5
    edge_color = ["g" if d["local"] == 1 else "grey" for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=width, edge_color=edge_color, connectionstyle=connectstyle)

    labels = {tuple(edge): f"{attrs['weight']}" for *edge, attrs in G.edges(keys=True, data=True)}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectstyle,
        label_pos=0.5,
        font_color="k",
        bbox={"alpha": 0},
    )


# -


# Create a mesh and function space. The function space will build an
# `IndexMap` for the degree-of-freedom map. The`IndexMap` describes how
# the degrees-of-freedom are distributed in parallel (across MPI ranks).
# From information on the parallel distribution we will be able to
# compute the communication graph.

# +
msh = mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=[(0.0, 0.0, 0.0), (2.0, 1.0, 1.0)],
    n=(22, 36, 19),
    cell_type=mesh.CellType.tetrahedron,
)
V = fem.functionspace(msh, ("Lagrange", 2))
# -

# An index map has the method `comm_graph` which can build a
# communication graph that represents data begin sent from the owning
# rank to ranks that ghost the data. We use the degree-of-freedom map
# IndexMap. Building the communication data is collective across MPI
# ranks. However, a non-empty graph is returned only on rank 0.

# +
comm_graph = V.dofmap.index_map.comm_graph()
# -

# A function for printing some communication graph metrics:


# +
def print_stats(G):
    print("Communication graph data:")
    print(f"  Num edges: {G.size()}")
    print(f"  Num local: {G.size('local')}")
    print(f"  Edges weight sum: {G.size('weight')}")
    if G.order() > 0:
        print(f"  Average edges per node: {G.size() / G.order()}")
    if G.size() > 0:
        print(f"  Average edge weight: {G.size('weight') / G.size()}")


# +

# The graph data will be processed on rank 0.

# +
if msh.comm.rank == 0:
    # To create a NetworkX directed graph, we get the graph data in a
    # form from which NetworkX can build a graph. Each edge will have a
    # weight and a 'local(1)/remote(0)' memory indicator.
    graph_data, node_weights = IndexMap.comm_graph_data(comm_graph)

    # Create a NetworkX directed graph.
    H = nx.MultiDiGraph()
    H.add_edges_from(graph_data)

    # Create graph with sorted nodes. This can be helpful for
    # visualisations.
    G = nx.MultiDiGraph()
    G.add_nodes_from(sorted(H.nodes(data=True)))
    G.add_edges_from(H.edges(data=True))

    print_stats(G)
    plot(G)
    plt.show()

    # Get graph data as a JSON string (useful if running from C++, in
    # which case the JSON string can be written to file)
    data_json_str = IndexMap.comm_to_json(comm_graph)
    data_json = json.loads(data_json_str)
    H1 = nx.adjacency_graph(data_json)

    # Create graph with sorted nodes. This can be helpful for
    # visualisations.
    G1 = nx.MultiDiGraph()
    G1.add_nodes_from(sorted(H1.nodes(data=True)))
    G1.add_edges_from(H1.edges(data=True))
    print_stats(G1)
# -
