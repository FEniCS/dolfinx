# Copyright (C) 2012-2019 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path

import numpy as np
import pytest
from dolfinx.fem import (Function, FunctionSpace, TensorFunctionSpace,
                         VectorFunctionSpace)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, create_unit_cube, create_unit_interval,
                          create_unit_square)
from mpi4py import MPI

# Supported XDMF file encoding
if MPI.COMM_WORLD.size > 1:
    encodings = [XDMFFile.Encoding.HDF5]
else:
    encodings = [XDMFFile.Encoding.HDF5, XDMFFile.Encoding.ASCII]

celltypes_2D = [CellType.triangle, CellType.quadrilateral]
celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


def mesh_factory(tdim, n):
    if tdim == 1:
        return create_unit_interval(MPI.COMM_WORLD, n)
    elif tdim == 2:
        return create_unit_square(MPI.COMM_WORLD, n, n)
    elif tdim == 3:
        return create_unit_cube(MPI.COMM_WORLD, n, n, n)


# --- Function


@pytest.mark.parametrize("use_pathlib", [True, False])
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_1d_scalar(tempdir, encoding, dtype, use_pathlib):
    xtype = np.real(dtype(0)).dtype
    filename2 = (Path(tempdir).joinpath("u1_.xdmf")if use_pathlib else Path(tempdir, "u1_.xdmf"))
    mesh = create_unit_interval(MPI.COMM_WORLD, 32, dtype=xtype)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V, dtype=dtype)
    u.x.array[:] = 1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0)

    with pytest.raises(RuntimeError):
        with XDMFFile(mesh.comm, filename2, "w", encoding=encoding) as file:
            file.write_mesh(mesh)
            file.write_function(u)

    V1 = FunctionSpace(mesh, ("Lagrange", 1))
    u1 = Function(V1, dtype=dtype)
    u1.interpolate(u)
    with XDMFFile(mesh.comm, filename2, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u1)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_2d_scalar(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "u2.xdmf")
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, cell_type, dtype=xtype)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V, dtype=dtype)
    u.x.array[:] = 1.0

    V1 = FunctionSpace(mesh, ("Lagrange", 1))
    u1 = Function(V1, dtype=dtype)
    u1.interpolate(u)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u1)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_3d_scalar(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "u3.xdmf")
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 3, 4, cell_type, dtype=xtype)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V, dtype=dtype)
    u.x.array[:] = 1.0

    V1 = FunctionSpace(mesh, ("Lagrange", 1))
    u1 = Function(V1, dtype=dtype)
    u1.interpolate(u)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u1)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_2d_vector(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "u_2dv.xdmf")
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 13, cell_type, dtype=xtype)
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V, dtype=dtype)
    u.x.array[:] = 1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0)

    V1 = VectorFunctionSpace(mesh, ("Lagrange", 1))
    u1 = Function(V1, dtype=dtype)
    u1.interpolate(u)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u1)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_3d_vector(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "u_3Dv.xdmf")
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type, dtype=xtype)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 1)), dtype=dtype)
    u.x.array[:] = 1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0)

    V1 = VectorFunctionSpace(mesh, ("Lagrange", 1))
    u1 = Function(V1, dtype=dtype)
    u1.interpolate(u)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u1)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_2d_tensor(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "tensor.xdmf")
    mesh = create_unit_square(MPI.COMM_WORLD, 16, 16, cell_type, dtype=xtype)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)), dtype=dtype)
    u.x.array[:] = 1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0)

    u1 = Function(TensorFunctionSpace(mesh, ("Lagrange", 1)), dtype=dtype)
    u1.interpolate(u)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u1)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_save_3d_tensor(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "u3t.xdmf")
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4, cell_type, dtype=xtype)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)), dtype=dtype)
    u.x.array[:] = 1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0)

    u1 = Function(TensorFunctionSpace(mesh, ("Lagrange", 1)), dtype=dtype)
    u1.interpolate(u)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u1)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_save_3d_vector_series(tempdir, encoding, dtype, cell_type):
    filename = Path(tempdir, "u_3D.xdmf")
    xtype = np.real(dtype(0)).dtype
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type, dtype=xtype)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 1)), dtype=dtype)

    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        u.x.array[:] = 1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0)
        file.write_function(u, 0.1)
        u.x.array[:] = 2.0 + (2j if np.issubdtype(dtype, np.complexfloating) else 0)
        file.write_function(u, 0.2)

    with XDMFFile(mesh.comm, filename, "a", encoding=encoding) as file:
        u.x.array[:] = 3.0 + (3j if np.issubdtype(dtype, np.complexfloating) else 0)
        file.write_function(u, 0.3)


def test_higher_order_function(tempdir):
    gmsh= pytest.importorskip('gmsh')
    # import gmsh
    from dolfinx.io import gmshio

    gmsh.initialize()

    # Choose if Gmsh output is verbose
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()

    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        # Using model.setCurrent(name) lets you change between models
        model.add("Sphere minus box")
        model.setCurrent("Sphere minus box")

        sphere_dim_tags = model.occ.addSphere(0, 0, 0, 1)
        box_dim_tags = model.occ.addBox(0, 0, 0, 1, 1, 1)
        model_dim_tags = model.occ.cut([(3, sphere_dim_tags)], [(3, box_dim_tags)])
        model.occ.synchronize()

        # Add physical tag 1 for exterior surfaces
        boundary = model.getBoundary(model_dim_tags[0], oriented=False)
        boundary_ids = [b[1] for b in boundary]
        model.addPhysicalGroup(2, boundary_ids, tag=1)
        model.setPhysicalName(2, 1, "Sphere surface")

        # Add physical tag 2 for the volume
        volume_entities = [model[1] for model in model.getEntities(3)]
        model.addPhysicalGroup(3, volume_entities, tag=2)
        model.setPhysicalName(3, 2, "Sphere volume")

        # Generate second order mesh and output gmsh messages to terminal
        model.mesh.generate(3)
        gmsh.option.setNumber("General.Terminal", 1)
        model.mesh.setOrder(2)
        gmsh.option.setNumber("General.Terminal", 0)

    msh, ct, ft = gmshio.model_to_mesh(model, mesh_comm, model_rank)
    msh.name = "ball_d2"
    ct.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_surface"

    u = Function(VectorFunctionSpace(msh, ("Lagrange", 1)))
    with pytest.raises(RuntimeError):
        filename = Path(tempdir, "u3D_P1.xdmf")
        with XDMFFile(msh.comm, filename, "w") as file:
            file.write_mesh(msh)
            file.write_function(u)

    u = Function(VectorFunctionSpace(msh, ("Lagrange", 2)))
    filename = Path(tempdir, "u3D_P2.xdmf")
    with XDMFFile(msh.comm, filename, "w") as file:
        file.write_mesh(msh)
        file.write_function(u)
