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

import numpy as np

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
adios2 = io.ADIOS2(config_path, msh.comm)
tag = "mesh-write"
adios2.add_io(filename="mesh.bp", tag=tag, mode="write")
# -

# +
io.write_mesh(adios2, tag, msh)

msh.geometry.x[:] += 4
io.write_mesh(adios2, tag, msh, 0.5)

msh.geometry.x[:] += 4
io.write_mesh(adios2, tag, msh, 1.0)

adios2.close(tag)
# -

# +
tag = "mesh-readrandomaccess"
adios2.add_io(filename="mesh.bp", tag=tag, engine_type="BP5", mode="readrandomaccess")
# -

# +
times = io.read_timestamps(adios2, tag)
print(f"Time stamps : {times}")
# -

# +
tag = "mesh-read"
adios2.add_io(filename="mesh.bp", tag=tag, engine_type="BP5", mode="read")
# -

# +
msh_read = io.read_mesh(adios2, tag, msh.comm)
print(np.max(msh_read.geometry.x))

io.update_mesh(adios2, tag, msh_read, 1)
print(np.max(msh_read.geometry.x))

io.update_mesh(adios2, tag, msh_read, 2)
print(np.max(msh_read.geometry.x))

adios2.close(tag)
# -

# +
tag = "meshtags-write"
adios2.add_io(filename="meshtags.bp", tag=tag, engine_type="BP5", mode="write")
# -

# +
for mt_name, mt in tags.items():
    io.write_meshtags(adios2, tag, msh, mt)

adios2.close(tag)
# -

# +
tag = "meshtags-read"
adios2.add_io(filename="meshtags.bp", tag=tag, engine_type="BP5", mode="read")
# -

# +
tags_read = io.read_meshtags(adios2, tag, msh)
adios2.close(tag)
# -
