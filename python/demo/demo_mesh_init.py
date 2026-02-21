# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Mesh threaded initialisationn

# +

from mpi4py import MPI

from dolfinx import mesh
from dolfinx.common import list_timings

# -


# +
msh = mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=[(0.0, 0.0, 0.0), (2.0, 1.0, 1.0)],
    n=(122, 136, 119),
    cell_type=mesh.CellType.tetrahedron,
)

msh.topology.create_entities(1, num_threads=4)
list_timings(MPI.COMM_WORLD)


# -
