# Copyright (C) 2019 Jorgen Dokken, Matthew Scroggs and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import os

import pytest
from mpi4py import MPI

from dolfinx import FunctionSpace, VectorFunctionSpace
from dolfinx.cpp.mesh import CellType
from dolfinx.io import XDMFFile
from dolfinx_utils.test.skips import skip_if_complex

from pipeline_tests import run_scalar_test, run_vector_test, run_dg_test


def get_mesh(cell_type, datadir):
    # In parallel, use larger meshes
    if cell_type == CellType.triangle:
        filename = "UnitSquareMesh_triangle.xdmf"
    elif cell_type == CellType.quadrilateral:
        filename = "UnitSquareMesh_quad.xdmf"
    elif cell_type == CellType.tetrahedron:
        filename = "UnitCubeMesh_tetra.xdmf"
    elif cell_type == CellType.hexahedron:
        filename = "UnitCubeMesh_hexahedron.xdmf"
    with XDMFFile(MPI.COMM_WORLD, os.path.join(datadir, filename), "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
        return xdmf.read_mesh(name="Grid")


parametrize_cell_types = pytest.mark.parametrize(
    "cell_type", [CellType.triangle, CellType.quadrilateral,
                  CellType.tetrahedron, CellType.hexahedron])
parametrize_cell_types_simplex = pytest.mark.parametrize(
    "cell_type", [CellType.triangle, CellType.tetrahedron])
parametrize_cell_types_tp = pytest.mark.parametrize(
    "cell_type", [CellType.quadrilateral, CellType.hexahedron])
parametrize_cell_types_quad = pytest.mark.parametrize(
    "cell_type", [CellType.quadrilateral])
parametrize_cell_types_hex = pytest.mark.parametrize(
    "cell_type", [CellType.hexahedron])


# Run tests on all spaces in periodic table on triangles and tetrahedra
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["Lagrange"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_P_simplex(family, degree, cell_type, datadir):
    if cell_type == CellType.tetrahedron and degree == 4:
        pytest.skip()  # Skip slowest test on tetrahedron

    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_scalar_test(mesh, V, degree)


@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["Lagrange"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_vector_P_simplex(family, degree, cell_type, datadir):
    if cell_type == CellType.tetrahedron and degree == 4:
        pytest.skip()  # Skip slowest test on tetrahedron

    mesh = get_mesh(cell_type, datadir)
    V = VectorFunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)


@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["DG"])
@pytest.mark.parametrize("degree", [2, 3])
def test_dP_simplex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_dg_test(mesh, V, degree)


@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["RT", "N1curl"])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_RT_N1curl_simplex(family, degree, cell_type, datadir):
    if cell_type == CellType.tetrahedron and degree == 4:
        pytest.skip()  # Skip slowest test on tetrahedron

    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree - 1)


@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["BDM", "N2curl"])
@pytest.mark.parametrize("degree", [1, 2])
def test_BDM_N2curl_simplex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)


# Skip slowest test in complex to stop CI timing out
@skip_if_complex
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["BDM", "N2curl"])
@pytest.mark.parametrize("degree", [3])
def test_BDM_N2curl_simplex_highest_order(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)


# Run tests on all spaces in periodic table on quadrilaterals and hexahedra
@parametrize_cell_types_tp
@pytest.mark.parametrize("family", ["Q"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_P_tp(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_scalar_test(mesh, V, degree)


@parametrize_cell_types_tp
@pytest.mark.parametrize("family", ["Q"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_vector_P_tp(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = VectorFunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)


@parametrize_cell_types_quad
@pytest.mark.parametrize("family", ["DQ", "DPC"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_dP_quad(family, degree, cell_type, datadir):
    if family == "DPC":
        pytest.skip()  # These space currently not implemented in basix

    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_dg_test(mesh, V, degree)


@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["DQ", "DPC"])
@pytest.mark.parametrize("degree", [1, 2])
def test_dP_hex(family, degree, cell_type, datadir):
    if family == "DPC":
        pytest.skip()  # These space currently not implemented in basix

    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_dg_test(mesh, V, degree)


@parametrize_cell_types_quad
@pytest.mark.parametrize("family", ["RTCE", "RTCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_RTC_quad(family, degree, cell_type, datadir):
    pytest.skip()  # These space currently not implemented in basix

    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree - 1)


@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["NCE", "NCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_NC_hex(family, degree, cell_type, datadir):
    pytest.skip()  # These space currently not implemented in basix

    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree - 1)


@parametrize_cell_types_quad
@pytest.mark.parametrize("family", ["BDMCE", "BDMCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_BDM_quad(family, degree, cell_type, datadir):
    pytest.skip()  # These space currently not implemented in basix

    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)


@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["AAE", "AAF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_AA_hex(family, degree, cell_type, datadir):
    pytest.skip()  # These space currently not implemented in basix
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)
