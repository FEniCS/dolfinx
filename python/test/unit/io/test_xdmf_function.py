# Copyright (C) 2012-2019 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest

import basix
from dolfinx.fem import Function, functionspace
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_cube, create_unit_interval, create_unit_square

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
    filename2 = Path(tempdir).joinpath("u1_.xdmf") if use_pathlib else Path(tempdir, "u1_.xdmf")
    mesh = create_unit_interval(MPI.COMM_WORLD, 32, dtype=xtype)
    V = functionspace(mesh, ("Lagrange", 2))
    u = Function(V, dtype=dtype)
    u.x.array[:] = 1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0)

    with pytest.raises(RuntimeError):
        with XDMFFile(mesh.comm, filename2, "w", encoding=encoding) as file:
            file.write_mesh(mesh)
            file.write_function(u)

    V1 = functionspace(mesh, ("Lagrange", 1))
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
    V = functionspace(mesh, ("Lagrange", 2))
    u = Function(V, dtype=dtype)
    u.x.array[:] = 1.0

    V1 = functionspace(mesh, ("Lagrange", 1))
    u1 = Function(V1, dtype=dtype)
    u1.interpolate(u)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u1)

    # Discontinuous (degree == 0)
    V = functionspace(mesh, ("Discontinuous Lagrange", 0))
    u = Function(V, dtype=dtype)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)

    # Discontinuous (degree > 0) should raise exception
    V = functionspace(mesh, ("Discontinuous Lagrange", 1))
    u = Function(V, dtype=dtype)
    with pytest.raises(RuntimeError):
        with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
            file.write_mesh(mesh)
            file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_3d_scalar(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "u3.xdmf")
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 3, 4, cell_type, dtype=xtype)
    V = functionspace(mesh, ("Lagrange", 2))
    u = Function(V, dtype=dtype)
    u.x.array[:] = 1.0

    V1 = functionspace(mesh, ("Lagrange", 1))
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
    gdim = mesh.geometry.dim

    V = functionspace(mesh, ("Lagrange", 2, (gdim,)))
    u = Function(V, dtype=dtype)
    u.x.array[:] = 1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0)

    V1 = functionspace(mesh, ("Lagrange", 1, (gdim,)))
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
    gdim = mesh.geometry.dim

    u = Function(functionspace(mesh, ("Lagrange", 1, (gdim,))), dtype=dtype)
    u.x.array[:] = 1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0)

    V1 = functionspace(mesh, ("Lagrange", 1, (gdim,)))
    u1 = Function(V1, dtype=dtype)
    u1.interpolate(u)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u1)

    V2 = functionspace(mesh, ("RT", 1))
    u2 = Function(V2, dtype=dtype)
    u2.interpolate(u)
    with pytest.raises(RuntimeError):
        with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
            file.write_mesh(mesh)
            file.write_function(u2)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_2d_tensor(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "tensor.xdmf")
    mesh = create_unit_square(MPI.COMM_WORLD, 16, 16, cell_type, dtype=xtype)
    gdim = mesh.geometry.dim

    u = Function(functionspace(mesh, ("Lagrange", 2, (gdim, gdim))), dtype=dtype)
    u.x.array[:] = 1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0)
    u1 = Function(functionspace(mesh, ("Lagrange", 1, (gdim, gdim))), dtype=dtype)
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

    gdim = mesh.geometry.dim
    u = Function(functionspace(mesh, ("Lagrange", 2, (gdim, gdim))), dtype=dtype)
    u.x.array[:] = 1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0)

    u1 = Function(functionspace(mesh, ("Lagrange", 1, (gdim, gdim))), dtype=dtype)
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
    gdim = mesh.geometry.dim
    u = Function(functionspace(mesh, ("Lagrange", 1, (gdim,))), dtype=dtype)
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
    """Test Function output for higher-order meshes."""
    gmsh = pytest.importorskip("gmsh")
    from dolfinx.io import gmshio

    gmsh.initialize()

    def gmsh_tet_model(order):
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model()
        comm = MPI.COMM_WORLD
        if comm.rank == 0:
            model.add("Sphere minus box")
            model.setCurrent("Sphere minus box")
            model.occ.addSphere(0, 0, 0, 1)
            model.occ.synchronize()
            volume_entities = [model[1] for model in model.getEntities(3)]
            model.addPhysicalGroup(3, volume_entities, tag=2)
            model.mesh.generate(3)
            gmsh.option.setNumber("General.Terminal", 1)
            model.mesh.setOrder(order)
            gmsh.option.setNumber("General.Terminal", 0)

        msh, _, _ = gmshio.model_to_mesh(model, comm, 0)
        return msh

    def gmsh_hex_model(order):
        model = gmsh.model()
        gmsh.option.setNumber("General.Terminal", 0)
        model.add("Hexahedral mesh")
        model.setCurrent("Hexahedral mesh")
        comm = MPI.COMM_WORLD
        if comm.rank == 0:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)
            circle = model.occ.addDisk(0, 0, 0, 1, 1)
            circle_inner = model.occ.addDisk(0, 0, 0, 0.5, 0.5)
            cut = model.occ.cut([(2, circle)], [(2, circle_inner)])[0]
            extruded_geometry = model.occ.extrude(cut, 0, 0, 0.5, numElements=[5], recombine=True)
            model.occ.synchronize()
            model.addPhysicalGroup(2, [cut[0][1]], tag=1)
            model.setPhysicalName(2, 1, "2D cylinder")
            boundary_entities = model.getEntities(2)
            other_boundary_entities = []
            for entity in boundary_entities:
                if entity != cut[0][1]:
                    other_boundary_entities.append(entity[1])
            model.addPhysicalGroup(2, other_boundary_entities, tag=3)
            model.setPhysicalName(2, 3, "Remaining boundaries")
            model.mesh.generate(3)
            model.mesh.setOrder(order)
            volume_entities = []
            for entity in extruded_geometry:
                if entity[0] == 3:
                    volume_entities.append(entity[1])
            model.addPhysicalGroup(3, volume_entities, tag=1)
            model.setPhysicalName(3, 1, "Mesh volume")

        msh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, 0)
        return msh

    # -- Degree 1 mesh (tet)
    msh = gmsh_tet_model(1)
    gdim = msh.geometry.dim
    assert msh.geometry.cmap.degree == 1

    # Write P1 Function
    u = Function(functionspace(msh, ("Lagrange", 1, (gdim,))))
    filename = Path(tempdir, "u3D_P1.xdmf")
    with XDMFFile(msh.comm, filename, "w") as file:
        file.write_mesh(msh)
        file.write_function(u)

    # Write P2 Function (exception expected)
    u = Function(functionspace(msh, ("Lagrange", 2, (gdim,))))
    filename = Path(tempdir, "u3D_P2.xdmf")
    with pytest.raises(RuntimeError):
        with XDMFFile(msh.comm, filename, "w") as file:
            file.write_mesh(msh)
            file.write_function(u)

    # -- Degree 2 mesh (tet)
    msh = gmsh_tet_model(2)
    gdim = msh.geometry.dim
    assert msh.geometry.cmap.degree == 2

    # Write P1 Function (exception expected)
    u = Function(functionspace(msh, ("Lagrange", 1, (gdim,))))
    with pytest.raises(RuntimeError):
        filename = Path(tempdir, "u3D_P1.xdmf")
        with XDMFFile(msh.comm, filename, "w") as file:
            file.write_mesh(msh)
            file.write_function(u)

    # Write P2 Function
    u = Function(functionspace(msh, ("Lagrange", 2, (gdim,))))
    filename = Path(tempdir, "u3D_P2.xdmf")
    with XDMFFile(msh.comm, filename, "w") as file:
        file.write_mesh(msh)
        file.write_function(u)

    # -- Degree 3 mesh (tet)
    # NOTE: XDMF/ParaView does not support TETRAHEDRON_20
    msh = gmsh_tet_model(3)
    gdim = msh.geometry.dim
    assert msh.geometry.cmap.degree == 3

    # Write P2 Function (exception expected)
    u = Function(functionspace(msh, ("Lagrange", 2, (gdim,))))
    with pytest.raises(RuntimeError):
        filename = Path(tempdir, "u3D_P3.xdmf")
        with XDMFFile(msh.comm, filename, "w") as file:
            file.write_mesh(msh)
            file.write_function(u)

    # Write P3 GLL Function (exception expected)
    ufl_e = basix.ufl.element(
        basix.ElementFamily.P, basix.CellType.tetrahedron, 3, basix.LagrangeVariant.gll_warped
    )
    u = Function(functionspace(msh, ufl_e))
    with pytest.raises(RuntimeError):
        filename = Path(tempdir, "u3D_P3.xdmf")
        with XDMFFile(msh.comm, filename, "w") as file:
            file.write_mesh(msh)
            file.write_function(u)

    # Write P3 equispaced Function
    ufl_e = basix.ufl.element(
        basix.ElementFamily.P, basix.CellType.tetrahedron, 3, basix.LagrangeVariant.equispaced
    )
    u1 = Function(functionspace(msh, ufl_e))
    u1.interpolate(u)
    filename = Path(tempdir, "u3D_P3.xdmf")
    with XDMFFile(msh.comm, filename, "w") as file:
        file.write_mesh(msh)
        file.write_function(u1)

    # --  Degree 2 mesh (hex)
    msh = gmsh_hex_model(2)
    gdim = msh.geometry.dim
    assert msh.geometry.cmap.degree == 2

    # Write Q1 Function (exception expected)
    u = Function(functionspace(msh, ("Lagrange", 1, (gdim,))))
    with pytest.raises(RuntimeError):
        filename = Path(tempdir, "u3D_Q1.xdmf")
        with XDMFFile(msh.comm, filename, "w") as file:
            file.write_mesh(msh)
            file.write_function(u)

    # Write Q2 Function
    u = Function(functionspace(msh, ("Lagrange", 2, (gdim,))))
    filename = Path(tempdir, "u3D_Q2.xdmf")
    with XDMFFile(msh.comm, filename, "w") as file:
        file.write_mesh(msh)
        file.write_function(u)

    # TODO: Higher-order gmsh hex meshes not yet supported by DOLFINx
    #
    # # Degree 3 mesh (hex)
    # msh = gmsh_hex_model(3)
    # assert msh.geometry.cmap.degree == 3
    # gdim = msh.geometry.dim
    # u = Function(functionspace(msh, ("Lagrange", 1, (gdim,))))
    # with pytest.raises(RuntimeError):
    #     filename = Path(tempdir, "u3D_Q1.xdmf")
    #     with XDMFFile(msh.comm, filename, "w") as file:
    #         file.write_mesh(msh)
    #         file.write_function(u)
    # u = Function(functionspace(msh, ("Lagrange", 2, (gdim,))))
    # filename = Path(tempdir, "u3D_Q2.xdmf")
    # with XDMFFile(msh.comm, filename, "w") as file:
    #     file.write_mesh(msh)
    #     file.write_function(u)

    gmsh.finalize()
