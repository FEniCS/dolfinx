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
    n=(8, 8),
    cell_type=mesh.CellType.triangle,
)
# -

# Create meshtags of all entity types for the above mesh

# +
tags = {}
for dim in range(msh.topology.dim + 1):
    msh.topology.create_connectivity(dim, msh.topology.dim)
    imap = msh.topology.index_map(dim)
    num_entities_local = imap.size_local
    entities = np.arange(num_entities_local, dtype=np.int32)
    values = imap.local_range[0] + entities
    mt = mesh.meshtags(msh, dim, entities, values)
    mt.name = f"entity_{dim}"
    tags[mt.name] = mt

# -

# +
config_path = os.getcwd() + "/checkpointing.yml"
adios2 = io.ADIOS2(config_path, msh.comm, filename="mesh.bp", tag="mesh-write", mode="write")
# -

# +
io.write_mesh(adios2, msh)
for mt_name, mt in tags.items():
    io.write_meshtags(adios2, msh, mt)

adios2.close()
# -

# +
adios2_read = io.ADIOS2(
    msh.comm, filename="mesh.bp", tag="mesh-read", engine_type="BP5", mode="read"
)
# -

# +
msh_read = io.read_mesh(adios2_read, msh.comm)
tags_read = io.read_meshtags(adios2_read, msh_read)
adios2_read.close()
# -

# +
adios2_write = io.ADIOS2(
    msh_read.comm, filename="mesh2.bp", tag="mesh-write", engine_type="BP5", mode="write"
)
# -

# +
io.write_mesh(adios2_write, msh_read)

for mt_name, mt_read in tags_read.items():
    io.write_meshtags(adios2_write, msh_read, mt_read)


adios2_write.close()
# -
