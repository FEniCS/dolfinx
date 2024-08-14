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
import importlib.util
import os

if importlib.util.find_spec("adios2") is not None:
    import dolfinx

    if not dolfinx.has_adios2:
        print("This demo requires DOLFINx to be compiled with ADIOS2 enabled.")
        exit(0)
else:
    print("This demo requires ADIOS2.")
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
adios2 = io.ADIOS2(config_path, msh.comm, filename="mesh.bp", tag="mesh-write", mode="write")
# -

# +
io.write_mesh(adios2, msh)
adios2.close()
# -

# +
adios2_read = io.ADIOS2(
    msh.comm, filename="mesh.bp", tag="mesh-read", engine_type="BP5", mode="read"
)
# -

# +
msh_read = io.read_mesh(adios2_read, msh.comm)
# -

# +
print(type(msh_read))
print(msh_read.name)
print(msh_read.geometry.x.dtype)
print(msh_read.geometry.x)
# -
