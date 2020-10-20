from IPython import embed
import h5py
import argparse

import dolfinx
import dolfinx.io
import numpy as np
import ufl
from mpi4py import MPI
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
#                          "mesh2.xdmf",
#                          "r") as xdmf:
#     with dolfinx.common.Timer("~ Read mesh (old)"):
#         mesh = xdmf.read_mesh(name="mesh")
#         mesh.topology.create_connectivity(
#             mesh.topology.dim-1, mesh.topology.dim)
# #     with dolfinx.common.Timer("~ Read cell marker (old)"):
# #         ct = xdmf.read_meshtags(
# #             mesh, "mesh")
#     with dolfinx.common.Timer("~ Read facet marker (old)"):
#         ft = xdmf.read_meshtags(mesh, "triangle mesh")

import h5py


infile = h5py.File("mesh.h5", "r+")
cf = infile["MeshFunction"]
# Reshape values in mesh function (For cells)
cf_val = cf["1"]["values"]
values = cf_val[()]
new_values = values.reshape((values.shape[0], 1))
cf["1"].pop("values")
cf["1"].create_dataset("values", data=new_values)
# Reshape values in mesh function (For facets)
cf_val = cf["0"]["values"]
values = cf_val[()]
new_values = values.reshape((values.shape[0], 1))
cf["0"].pop("values")
cf["0"].create_dataset("values", data=new_values)
infile.close()

with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                         "mesh.xdmf",
                         "r") as xdmf:
    with dolfinx.common.Timer("~ Read mesh (old)"):
        mesh = xdmf.read_mesh(name="mesh")
        mesh.topology.create_connectivity(
            mesh.topology.dim-1, mesh.topology.dim)
    with dolfinx.common.Timer("~ Read cell marker (old)"):
        ct = xdmf.read_meshtags(
            mesh, "mesh")

    with dolfinx.common.Timer("~ Read facet marker (old)"):
        ft = xdmf.read_meshtags(mesh, "mesh")
