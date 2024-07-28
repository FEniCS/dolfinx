# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Checkpointing
#
# This demo is implemented in {download}`demo_checkpointing.py`. It
# illustrates checkpointing using ADIOS2:
#

# +
try:
    import adios2.bindings
except:
    print("This demo requires adios2.bindings.")
    exit(0)

from mpi4py import MPI

import numpy as np

from dolfinx import io, mesh
# -

# Note that it is important to first `from mpi4py import MPI` to
# ensure that MPI is correctly initialised.

# We create a rectangular {py:class}`Mesh <dolfinx.mesh.Mesh>` using
# {py:func}`create_rectangle <dolfinx.mesh.create_rectangle>`, and
# save it to a file `mesh.bp`

# +
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(16, 16),
    cell_type=mesh.CellType.triangle,
)
# -

# +
import adios2.bindings
filename = "mesh.bp"
engine_type = "BP5"
adios = adios2.bindings.ADIOS(msh.comm)
adios2_io = adios.DeclareIO("mesh-write")
adios2_io.SetEngine(engine_type)
adios2_engine = adios2_io.Open(filename, adios2.bindings.Mode.Write)
# -

# +
io.write_mesh(adios2_io, adios2_engine, msh)
adios2_engine.Close()
# -
