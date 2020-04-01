# Copyright (C) 2019 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for custom Python assemblers"""

import ctypes
import ctypes.util
import importlib
import math
import os
import pathlib
import time

import cffi
import numba
import numba.cffi_support
import numpy as np
import petsc4py.lib
import pytest
from petsc4py import PETSc
from petsc4py import get_config as PETSc_get_config

import dolfinx
import ufl
from ufl import dx, inner

# Get details of PETSc install
petsc_dir = PETSc_get_config()['PETSC_DIR']
petsc_arch = petsc4py.lib.getPathArchPETSc()[1]


# Get PETSc int and scalar types
if np.dtype(PETSc.ScalarType).kind == 'c':
    complex = True
else:
    complex = False

scalar_size = np.dtype(PETSc.ScalarType).itemsize
index_size = np.dtype(PETSc.IntType).itemsize

if index_size == 8:
    c_int_t = "int64_t"
    ctypes_index = ctypes.c_int64
elif index_size == 4:
    c_int_t = "int32_t"
    ctypes_index = ctypes.c_int32
else:
    raise RecursionError("Unknown PETSc index type.")

if complex and scalar_size == 16:
    c_scalar_t = "double _Complex"
    numba_scalar_t = numba.types.complex128
elif complex and scalar_size == 8:
    c_scalar_t = "float _Complex"
    numba_scalar_t = numba.types.complex64
elif not complex and scalar_size == 8:
    c_scalar_t = "double"
    numba_scalar_t = numba.types.float64
elif not complex and scalar_size == 4:
    c_scalar_t = "float"
    numba_scalar_t = numba.types.float32
else:
    raise RuntimeError(
        "Cannot translate PETSc scalar type to a C type, complex: {} size: {}.".format(complex, scalar_size))


# Load PETSc library via ctypes
petsc_lib_name = ctypes.util.find_library("petsc")
if petsc_lib_name is not None:
    petsc_lib_ctypes = ctypes.CDLL(petsc_lib_name)
else:
    try:
        petsc_lib_ctypes = ctypes.CDLL(os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.so"))
    except OSError:
        petsc_lib_ctypes = ctypes.CDLL(os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.dylib"))
    except OSError:
        print("Could not load PETSc library for CFFI (ABI mode).")
        raise

# Get the PETSc MatSetValuesLocal function via ctypes
MatSetValues_ctypes = petsc_lib_ctypes.MatSetValuesLocal
MatSetValues_ctypes.argtypes = (ctypes.c_void_p, ctypes_index, ctypes.POINTER(
    ctypes_index), ctypes_index, ctypes.POINTER(ctypes_index), ctypes.c_void_p, ctypes.c_int)
del petsc_lib_ctypes


ADD_VALUES = PETSc.InsertMode.ADD_VALUES


# CFFI - register complex types
ffi = cffi.FFI()
numba.cffi_support.register_type(ffi.typeof('double _Complex'), numba.types.complex128)
numba.cffi_support.register_type(ffi.typeof('float _Complex'), numba.types.complex64)

# Get MatSetValuesLocal from PETSc available via cffi in ABI mode
ffi.cdef("""int MatSetValuesLocal(void* mat, {0} nrow, const {0}* irow,
                                  {0} ncol, const {0}* icol, const {1}* y, int addv);
""".format(c_int_t, c_scalar_t))


if petsc_lib_name is not None:
    petsc_lib_cffi = ffi.dlopen(petsc_lib_name)
else:
    try:
        petsc_lib_cffi = ffi.dlopen(os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.so"))
    except OSError:
        petsc_lib_cffi = ffi.dlopen(os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.dylib"))
    except OSError:
        print("Could not load PETSc library for CFFI (ABI mode).")
        raise
MatSetValues_abi = petsc_lib_cffi.MatSetValuesLocal

# Make MatSetValuesLocal from PETSc available via cffi in API mode
worker = os.getenv('PYTEST_XDIST_WORKER', None)
module_name = "_petsc_cffi_{}".format(worker)
if dolfinx.MPI.comm_world.Get_rank() == 0:
    os.environ["CC"] = "mpicc"
    ffibuilder = cffi.FFI()
    ffibuilder.cdef("""
        typedef int... PetscInt;
        typedef ... PetscScalar;
        typedef int... InsertMode;
        int MatSetValuesLocal(void* mat, PetscInt nrow, const PetscInt* irow,
                                PetscInt ncol, const PetscInt* icol,
                                const PetscScalar* y, InsertMode addv);
    """)
    ffibuilder.set_source(module_name, """
        # include "petscmat.h"
    """,
                          libraries=['petsc'],
                          include_dirs=[os.path.join(petsc_dir, petsc_arch, 'include'),
                                        os.path.join(petsc_dir, 'include')],
                          library_dirs=[os.path.join(petsc_dir, petsc_arch, 'lib')],
                          extra_compile_args=[])

    # Build module in same directory as test file
    path = pathlib.Path(__file__).parent.absolute()
    ffibuilder.compile(tmpdir=path, verbose=False)

dolfinx.MPI.comm_world.barrier()

spec = importlib.util.find_spec(module_name)
if spec is None:
    raise ImportError("Failed to find CFFI generated module")
module = importlib.util.module_from_spec(spec)

numba.cffi_support.register_module(module)
MatSetValues_api = module.lib.MatSetValuesLocal
numba.cffi_support.register_type(module.ffi.typeof("PetscScalar"), numba_scalar_t)


# See https://github.com/numba/numba/issues/4036 for why we need 'sink'
@numba.njit
def sink(*args):
    pass


@numba.njit
def area(x0, x1, x2) -> float:
    """Compute the area of a triangle embedded in 2D from the three vertices"""
    a = (x1[0] - x2[0])**2 + (x1[1] - x2[1])**2
    b = (x0[0] - x2[0])**2 + (x0[1] - x2[1])**2
    c = (x0[0] - x1[0])**2 + (x0[1] - x1[1])**2
    return math.sqrt(2 * (a * b + a * c + b * c) - (a**2 + b**2 + c**2)) / 4.0


@numba.njit
def assemble_vector(b, mesh, dofmap):
    """Assemble simple linear form over a mesh into the array b"""
    pos, x_dofmap, x = mesh
    q0, q1 = 1 / 3.0, 1 / 3.0
    for i, cell in enumerate(pos[:-1]):
        num_vertices = pos[i + 1] - pos[i]
        # FIXME: This assumes a particular geometry dof layout
        c = x_dofmap[cell:cell + num_vertices]
        A = area(x[c[0]], x[c[1]], x[c[2]])
        b[dofmap[i * 3 + 0]] += A * (1.0 - q0 - q1)
        b[dofmap[i * 3 + 1]] += A * q0
        b[dofmap[i * 3 + 2]] += A * q1


@numba.njit
def assemble_vector_ufc(b, kernel, mesh, dofmap):
    """Assemble provided FFCX/UFC kernel over a mesh into the array b"""
    pos, x_dofmap, x = mesh
    entity_local_index = np.array([0], dtype=np.int32)
    perm = np.array([0], dtype=np.uint8)
    geometry = np.zeros((3, 2))
    coeffs = np.zeros(1, dtype=PETSc.ScalarType)
    constants = np.zeros(1, dtype=PETSc.ScalarType)

    b_local = np.zeros(3, dtype=PETSc.ScalarType)
    for i, cell in enumerate(pos[:-1]):
        num_vertices = pos[i + 1] - pos[i]
        # FIXME: This assumes a particular geometry dof layout
        c = x_dofmap[cell:cell + num_vertices]
        for j in range(3):
            for k in range(2):
                geometry[j, k] = x[c[j], k]
        b_local.fill(0.0)
        kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs),
               ffi.from_buffer(constants),
               ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index),
               ffi.from_buffer(perm), 0)
        for j in range(3):
            b[dofmap[i * 3 + j]] += b_local[j]


@numba.njit(fastmath=True)
def assemble_matrix_cffi(A, mesh, dofmap, set_vals, mode):
    """Assemble P1 mass matrix over a mesh into the PETSc matrix A"""

    # Mesh data
    cell_ptr, x_dofmap, x = mesh

    # Quadrature points and weights
    q = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], dtype=np.double)
    weights = np.full(3, 1.0 / 3.0, dtype=np.double)

    # Loop over cells
    N = np.empty(3, dtype=np.double)
    A_local = np.empty((3, 3), dtype=PETSc.ScalarType)
    for i, cell in enumerate(cell_ptr[:-1]):
        num_vertices = cell_ptr[i + 1] - cell_ptr[i]
        c = x_dofmap[cell:cell + num_vertices]
        cell_area = area(x[c[0]], x[c[1]], x[c[2]])

        # Loop over quadrature points
        A_local[:] = 0.0
        for j in range(q.shape[0]):
            N[0], N[1], N[2] = 1.0 - q[j, 0] - q[j, 1], q[j, 0], q[j, 1]
            for k in range(3):
                for l in range(3):
                    A_local[k, l] += weights[j] * cell_area * N[k] * N[l]

        # Add to global tensor
        pos = dofmap[3 * i:3 * i + 3]
        set_vals(A, 3, ffi.from_buffer(pos), 3, ffi.from_buffer(pos), ffi.from_buffer(A_local), mode)
    sink(A_local, dofmap)


@numba.njit
def assemble_matrix_ctypes(A, mesh, dofmap, set_vals, mode):
    """Assemble P1 mass matrix over a mesh into the PETSc matrix A"""
    cell_ptr, x_dofmap, x = mesh
    q = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], dtype=np.double)
    weights = np.full(3, 1.0 / 3.0, dtype=np.double)

    # Loop over cells
    N = np.empty(3, dtype=np.double)
    A_local = np.empty((3, 3), dtype=PETSc.ScalarType)
    for i, cell in enumerate(cell_ptr[:-1]):
        num_vertices = cell_ptr[i + 1] - cell_ptr[i]
        # FIXME: This assumes a particular geometry dof layout
        c = x_dofmap[cell:cell + num_vertices]
        cell_area = area(x[c[0]], x[c[1]], x[c[2]])

        # Loop over quadrature points
        A_local[:] = 0.0
        for j in range(q.shape[0]):
            N[0], N[1], N[2] = 1.0 - q[j, 0] - q[j, 1], q[j, 0], q[j, 1]
            for k in range(3):
                for l in range(3):
                    A_local[k, l] += weights[j] * cell_area * N[k] * N[l]

        rows = cols = dofmap[3 * i:3 * i + 3]
        set_vals(A, 3, rows.ctypes, 3, cols.ctypes, A_local.ctypes, mode)


@numba.njit
def assemble_exterior_facets_cffi(A, kernel, mesh, coeffs, consts, perm, dofmap,
                                  num_dofs_per_element, gdim, facet_info, set_vals, mode):
    """Assemble exterior facet integral into a PETSc matrix A"""

    # Mesh data
    pos, x_dofmap, x = mesh
    geometry = np.zeros((pos[1] - pos[0], gdim))

    facet_index = np.zeros(1, dtype=np.int32)
    facet_perm = np.zeros(1, dtype=np.uint8)

    A_local = np.zeros((num_dofs_per_element, num_dofs_per_element),
                       dtype=PETSc.ScalarType)
    # Permutation info
    cell_perms, facet_perms = perm
    for i in range(facet_info.shape[0]):
        cell_index, local_facet = facet_info[i]
        cell = pos[cell_index]
        facet_index[0] = local_facet
        num_vertices = pos[cell_index + 1] - pos[cell_index]
        # FIXME: This assumes a particular geometry dof layout
        c = x_dofmap[cell:cell + num_vertices]
        for j in range(num_vertices):
            for k in range(gdim):
                geometry[j, k] = x[c[j], k]
        A_local.fill(0.0)
        facet_perm[0] = facet_perms[local_facet, cell_index]
        kernel(ffi.from_buffer(A_local),
               ffi.from_buffer(coeffs[cell_index, :]),
               ffi.from_buffer(consts),
               ffi.from_buffer(geometry),
               ffi.from_buffer(facet_index),
               ffi.from_buffer(facet_perm),
               cell_perms[cell_index])

        local_pos = dofmap[num_dofs_per_element * cell_index:
                           num_dofs_per_element * cell_index
                           + num_dofs_per_element]

        ierr_loc = set_vals(A, num_dofs_per_element, ffi.from_buffer(local_pos),
                            num_dofs_per_element, ffi.from_buffer(local_pos),
                            ffi.from_buffer(A_local), mode)
        if ierr_loc != 0:
            raise ValueError("Assembly failed")
    sink(A_local, local_pos)


def test_custom_mesh_loop_rank1():

    # Create mesh and function space
    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 64, 64)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))

    # Unpack mesh and dofmap data
    pos = mesh.geometry.dofmap().offsets()
    x_dofs = mesh.geometry.dofmap().array()
    x = mesh.geometry.x
    dofs = V.dofmap.list.array()

    # Assemble with pure Numba function (two passes, first will include JIT overhead)
    b0 = dolfinx.Function(V)
    for i in range(2):
        with b0.vector.localForm() as b:
            b.set(0.0)
            start = time.time()
            assemble_vector(np.asarray(b), (pos, x_dofs, x), dofs)
            end = time.time()
            print("Time (numba, pass {}): {}".format(i, end - start))

    b0.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert(b0.vector.sum() == pytest.approx(1.0))

    # Test against generated code and general assembler
    v = ufl.TestFunction(V)
    L = inner(1.0, v) * dx

    start = time.time()
    b1 = dolfinx.fem.assemble_vector(L)
    end = time.time()
    print("Time (C++, pass 1):", end - start)

    with b1.localForm() as b_local:
        b_local.set(0.0)
    start = time.time()
    dolfinx.fem.assemble_vector(b1, L)
    end = time.time()
    print("Time (C++, pass 2):", end - start)

    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert((b1 - b0.vector).norm() == pytest.approx(0.0))

    # Assemble using generated tabulate_tensor kernel and Numba assembler
    b3 = dolfinx.Function(V)
    ufc_form = dolfinx.jit.ffcx_jit(L)
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    for i in range(2):
        with b3.vector.localForm() as b:
            b.set(0.0)
            start = time.time()
            assemble_vector_ufc(np.asarray(b), kernel, (pos, x_dofs, x), dofs)
            end = time.time()
            print("Time (numba/cffi, pass {}): {}".format(i, end - start))

    b3.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert((b3.vector - b0.vector).norm() == pytest.approx(0.0))


def test_custom_mesh_loop_ctypes_rank2():
    """Test numba assembler for bilinear form"""

    # Create mesh and function space
    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 64, 64)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))

    # Extract mesh and dofmap data
    pos = mesh.geometry.dofmap().offsets()
    x_dofs = mesh.geometry.dofmap().array()
    x = mesh.geometry.x
    dofs = V.dofmap.list.array()

    # Generated case with general assembler
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx
    A0 = dolfinx.fem.assemble_matrix(a)
    A0.assemble()
    A0.zeroEntries()

    start = time.time()
    dolfinx.fem.assemble_matrix(A0, a)
    end = time.time()
    print("Time (C++, pass 2):", end - start)
    A0.assemble()

    # Custom case
    A1 = A0.copy()
    for i in range(2):
        A1.zeroEntries()
        mat = A1.handle
        start = time.time()
        assemble_matrix_ctypes(mat, (pos, x_dofs, x), dofs, MatSetValues_ctypes, PETSc.InsertMode.ADD_VALUES)
        end = time.time()
        print("Time (numba, pass {}): {}".format(i, end - start))
        A1.assemble()

    assert (A0 - A1).norm() == pytest.approx(0.0, abs=1.0e-9)


@pytest.mark.parametrize("set_vals", [MatSetValues_abi, MatSetValues_api])
def test_custom_mesh_loop_cffi_rank2(set_vals):
    """Test numba assembler for bilinear form"""

    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 64, 64)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))

    # Test against generated code and general assembler
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx
    A0 = dolfinx.fem.assemble_matrix(a)
    A0.assemble()

    A0.zeroEntries()
    start = time.time()
    dolfinx.fem.assemble_matrix(A0, a)
    end = time.time()
    print("Time (C++, pass 2):", end - start)
    A0.assemble()

    # Unpack mesh and dofmap data
    pos = mesh.geometry.dofmap().offsets()
    x_dofs = mesh.geometry.dofmap().array()
    x = mesh.geometry.x
    dofs = V.dofmap.list.array()

    A1 = A0.copy()
    for i in range(2):
        A1.zeroEntries()
        start = time.time()
        assemble_matrix_cffi(A1.handle, (pos, x_dofs, x), dofs, set_vals, PETSc.InsertMode.ADD_VALUES)
        end = time.time()
        print("Time (Numba, pass {}): {}".format(i, end - start))
        A1.assemble()

    assert (A1 - A0).norm() == pytest.approx(0.0)


@pytest.mark.parametrize("set_vals", [MatSetValues_abi, MatSetValues_api])
def test_exterior_facet_cffi(set_vals):
    """Test numba assembler for bilinear form"""

    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 64, 64)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))

    def top(x):
        return np.isclose(x[1], 1)
    fdim = mesh.topology.dim - 1
    top_facets = dolfinx.mesh.locate_entities_geometrical(mesh, 1, top,
                                                          boundary_only=True)
    mt = dolfinx.mesh.MeshTags(mesh, fdim, top_facets, 3)

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=3)

    # Test against generated code and general assembler
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a1 = -3 * inner(u, v) * ds
    a2 = dolfinx.Constant(mesh, 2) * inner(u, v) * ufl.ds
    a = a1 + a2
    A0 = dolfinx.fem.assemble_matrix(a)
    A0.assemble()
    A0.zeroEntries()

    start = time.time()
    dolfinx.fem.assemble_matrix(A0, a)
    end = time.time()
    print("Time (C++, pass 2):", end - start)
    A0.assemble()

    # Unpack mesh and dofmap data
    pos = V.mesh.geometry.dofmap().offsets()
    x_dofs = V.mesh.geometry.dofmap().array()
    x = V.mesh.geometry.x
    dofs = V.dofmap.list.array()
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs
    gdim = mesh.geometry.dim

    # Get cell orientation data
    permutation_info = V.mesh.topology.get_cell_permutation_info()
    facet_permutation_info = V.mesh.topology.get_facet_permutations()
    perm = (permutation_info, facet_permutation_info)

    # Create various forms
    cpp_form = dolfinx.Form(a)._cpp_object
    ufc_form = dolfinx.jit.ffcx_jit(a)

    # Get coefficients and constants
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    form_consts = dolfinx.cpp.fem.pack_constants(cpp_form)

    # Get subdomain ids for the exterior facet integrals
    formintegral = cpp_form.integrals()
    subdomain_ids = formintegral.integral_ids(dolfinx.cpp.fem.FormIntegrals.Type.exterior_facet)
    num_exterior_integrals = len(subdomain_ids)

    A1 = A0.copy()
    for i in range(2):
        A1.zeroEntries()
        start = time.time()
        for j in range(num_exterior_integrals):
            facet_info = dolfinx.cpp.fem.pack_exterior_facets(cpp_form, j)
            subdomain_id = subdomain_ids[j]
            facet_kernel = ufc_form.create_exterior_facet_integral(
                subdomain_id).tabulate_tensor

            assemble_exterior_facets_cffi(A1.handle, facet_kernel,
                                          (pos, x_dofs, x), form_coeffs, form_consts,
                                          perm, dofs, num_dofs_per_element, gdim, facet_info,
                                          set_vals, PETSc.InsertMode.ADD_VALUES)
            A1.assemble()

        end = time.time()
        print("Time (Numba, pass {}): {}".format(i, end - start))
        A1.assemble()

    assert (A1 - A0).norm() == pytest.approx(0.0)
