# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Parallel communication analysis
#

import itertools as it
import json

from mpi4py import MPI

import matplotlib.pyplot as plt
import networkx as nx

from dolfinx import fem, mesh

# Create a mesh and function space. The function space will create and
# IndexMap for the degree-of-freedom map, which includes information on
# parallel communication patterns.
msh = mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=[(0.0, 0.0, 0.0), (2.0, 1.0, 1.0)],
    n=(22, 36, 19),
    cell_type=mesh.CellType.tetrahedron,
)
V = fem.functionspace(msh, ("Lagrange", 2))

# Compute communication pattern graph data from the degree-of-freedom
# map IndexMap. Building the data is collective across MPI ranks.
# However, data is returned only on rank 0.
graph_data, node_weights = V.dofmap.index_map.comm_graph_data()
# g_data, xnode_weights = V.dofmap.index_map.comm_graph()

foo = V.dofmap.index_map.comm_graph_tojson()

if msh.comm.rank == 0:
    # Create a directed graph. Each edge will have a weight and a
    # 'local(1)/remote(0)' memory indicator.
    H = nx.MultiDiGraph()
    H.add_edges_from(graph_data)

    foox = nx.adjacency_data(H)
    print(foox)
    print("\n")
    print("ME:\n", foo)

    test1 = json.loads(foo)
    H = nx.adjacency_graph(test1)

    # Attach nodes weights (local size)
    # nodes = {p: {"weight": w} for p, w in enumerate(node_weights)}
    # nx.set_node_attributes(H, nodes)

    # Create graph with sorted nodes. This can be helpful for
    # visualisations.
    G = nx.MultiDiGraph()
    G.add_nodes_from(sorted(H.nodes(data=True)))
    G.add_edges_from(H.edges(data=True))

    # Print some communication graph metrics
    print("Communication graph data")
    print("  Num edges:", G.size())
    print("  Num local:", G.size("local"))
    print("  Edges weight sum:", G.size("weight"))
    print("  Average edges per node:", G.size() / G.order())
    print("  Average edge weight:", G.size("weight") / G.size())

    foo = nx.adjacency_data(G)
    print(foo)

    def plot(G):
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
        nx.draw_networkx_edges(
            G, pos, width=width, edge_color=edge_color, connectionstyle=connectstyle
        )

        labels = {
            tuple(edge): f"{attrs['weight']}" for *edge, attrs in G.edges(keys=True, data=True)
        }
        nx.draw_networkx_edge_labels(
            G,
            pos,
            labels,
            connectionstyle=connectstyle,
            label_pos=0.5,
            font_color="k",
            bbox={"alpha": 0},
        )

    plot(G)
    plt.show()
