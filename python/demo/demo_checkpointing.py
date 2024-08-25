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
import os

import dolfinx

if not dolfinx.common.has_adios2:
    print("This demo requires DOLFINx to be compiled with ADIOS2 enabled.")
    exit(0)

from mpi4py import MPI

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
    n=(8, 8),
    cell_type=mesh.CellType.triangle,
)
# -

# +
config_path = os.getcwd() + "/checkpointing.yml"
adios2 = io.ADIOS2(config_path, msh.comm)
tag = "mesh-write"
adios2.add_io(filename="mesh.bp", tag=tag, mode="write")
# -

# +
io.write_mesh(adios2, tag, msh)
adios2.close(tag)
# -

# # +
# adios2_read = io.ADIOS2(msh.comm)
# tag = "mesh-read"
# adios2_read.add_io(filename="mesh.bp", tag=tag, engine_type="BP5", mode="read")
# # -

# # +
# msh_read = io.read_mesh(adios2_read, tag, msh.comm)
# adios2_read.close(tag)
# # -

# +
# adios2_read = io.ADIOS2(msh.comm)
tag = "mesh-read"
adios2.add_io(filename="mesh.bp", tag=tag, engine_type="BP5", mode="read")
# -

# +
msh_read = io.read_mesh(adios2, tag, msh.comm)
adios2.close(tag)
# -
