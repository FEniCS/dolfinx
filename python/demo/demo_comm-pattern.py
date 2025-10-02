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
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_comm-pattern.py>`
# * {download}`Jupyter notebook <./demo_comm-pattern.ipynb>`
# ```
# This demo illustrates how to:
# - Build a graph that represents a parallel communication pattern
# - Analyse the parallel communication pattern using
#   [NetworkX](https://networkx.org/).
#
# The layout of a distributed array across processes (MPI ranks) is
# described in DOLFINx by an {py:class}`IndexMap
# <dolfinx.common.IndexMap>`. An {py:class}`IndexMap
# <dolfinx.common.IndexMap>` represents the range of locally 'owned'
# array indices and the indices that are ghosted on a rank. It also
# holds information on the ranks that the calling rank will send data to
# and ranks that will send data to the caller.
#

# +
import itertools as it
import json

from mpi4py import MPI

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.ticker import MaxNLocator

from dolfinx import fem, graph, mesh

# -


# The following function plots a directed graph, with the edge weights
# labeled. Each node is an MPI rank, and an edge represents a
# communication edge. The edge weights indicate the volume of data
# communicated.


# +
def plot_graph(G: nx.MultiGraph, egde_labels=False):
    """Plot the communication graph."""
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, alpha=0.75)
    nx.draw_networkx_labels(G, pos, font_size=12)

    width = 0.5
    edge_color = ["g" if d["local"] == 1 else "grey" for _, _, d in G.edges(data=True)]
    if egde_labels:
        # Curve edges to distinguish between in- and out-edges
        connectstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]

        # Color edges by local (shared memory) or remote (remote memory)
        # communication
        nx.draw_networkx_edges(
            G, pos, width=width, edge_color=edge_color, connectionstyle=connectstyle
        )

        labels = {tuple(edge): f"{attrs['weight']}" for *edge, attrs in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G,
            pos,
            labels,
            connectionstyle=connectstyle,
            label_pos=0.5,
            font_color="k",
            bbox={"alpha": 0},
        )
    else:
        nx.draw_networkx_edges(G, pos, width=width, edge_color=edge_color)


# -

# The following function produces bar charts with the number of out-edges
# per rank and the sum of the out edge weights (measure of data
# volume) per rank.


# +
def plot_bar(G: nx.MultiGraph):
    """Plot bars charts with the degree (number of 'out-edges') and the
    outward data volume for each rank.
    """

    ranks = range(G.order())
    num_edges = [len(nbrs) for _, nbrs in G.adj.items()]
    weights = [sum(data["weight"] for nbr, data in nbrs.items()) for _, nbrs in G.adj.items()]

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(ranks, num_edges)
    ax1.set_xlabel("rank")
    ax1.set_ylabel("out degree")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax2.bar(ranks, weights)
    ax2.set_xlabel("rank")
    ax2.set_ylabel("sum of edge weights")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))


# -

# Create a mesh and function space. The function space will build an
# {py:class}`IndexMap <dolfinx.common.IndexMap>` for the
# degree-of-freedom map. The {py:class}`IndexMap
# <dolfinx.common.IndexMap>` describes how the degrees-of-freedom are
# distributed in parallel (across MPI ranks). From information on the
# parallel distribution we will be able to compute the communication
# graph.

# +
msh = mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=[(0.0, 0.0, 0.0), (2.0, 1.0, 1.0)],
    n=(22, 36, 19),
    cell_type=mesh.CellType.tetrahedron,
)
V = fem.functionspace(msh, ("Lagrange", 2))
# -

# The function {py:func}`comm_graph <dolfinx.graph.comm_graph>` builds a
# communication graph that represents data begin sent from the owning
# rank to ranks that ghost the data. We use the degree-of-freedom map's
# `IndexMap`. Building the communication data is collective across MPI
# ranks. However, a non-empty graph is returned only on rank 0.

# +
comm_graph = graph.comm_graph(V.dofmap.index_map)
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


# -

# The graph data will be processed on rank 0. From the communication
# graph data, edge and node data for creating a `NetworkX`` graph is build
# using {py:fuc}`comm_graph_data <dolfinx.graph.comm_graph_data>`.
#
# Data for use with `NetworkX` can also be reconstructed from a JSON
# string. The JSON string can be created using {py:func}`comm_to_json
# <dolfinx.graph.comm_to_json>`. This is helpful for cases there a
# simulaton is executed and the graph data is written to file for later
# analysis.

# +
if msh.comm.rank == 0:
    # To create a NetworkX directed graph we build graph data in a form
    # from which we can create a NetworkX graph. Each edge will have a
    # weight and a 'local(1)/remote(0)' memory indicator and each node
    # has its local size and the number of ghosts.
    adj_data, node_data = graph.comm_graph_data(comm_graph)

    print("Test:", graph.comm_graph_data(comm_graph))

    # Create a NetworkX directed graph.
    H = nx.DiGraph()
    H.add_edges_from(adj_data)
    H.add_nodes_from(node_data)

    # Create graph with sorted nodes. This can be helpful for
    # visualisations.
    G = nx.DiGraph()
    G.add_nodes_from(sorted(H.nodes(data=True)))
    G.add_edges_from(H.edges(data=True))

    print_stats(G)

    plot_bar(G)
    plt.show()

    plot_graph(G, True)
    plt.show()

    # Get graph data as a JSON string (useful if running from C++, in
    # which case the JSON string can be written to file)
    data_json_str = graph.comm_to_json(comm_graph)
    H1 = nx.adjacency_graph(json.loads(data_json_str))

    # Create graph with sorted nodes. This can be helpful for
    # visualisations.
    G1 = nx.DiGraph()
    G1.add_nodes_from(sorted(H1.nodes(data=True)))
    G1.add_edges_from(H1.edges(data=True))
    print_stats(G1)

# -
