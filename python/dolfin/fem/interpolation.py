# -*- coding: utf-8 -*-
# Copyright (C) 2009-2011 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Interpolation of a :py:class:`Function <dolfin.functions.function.Function>` or
:py:class:`Expression <dolfin.functions.expression.Expression>` onto a
finite element space.

"""
import cffi
import numpy as np
import numba
import numba.cffi_support
from numba.typed import List
from petsc4py import PETSc

from dolfin import function, jit
import ffc
import ufl


def interpolate(v, V):
    """Return interpolation of a given function into a given finite
    element space.

    *Arguments*
        v
            a :py:class:`Function <dolfin.functions.function.Function>` or
            an :py:class:`Expression <dolfin.functions.expression.Expression>`
        V
            a :py:class:`FunctionSpace (standard, mixed, etc.)
            <dolfin.functions.functionspace.FunctionSpace>`

    *Example of usage*

        .. code-block:: python

            v = Expression("sin(pi*x[0])")
            V = FunctionSpace(mesh, "Lagrange", 1)
            Iv = interpolate(v, V)

    """

    # Check arguments
    # if not isinstance(V, cpp.functionFunctionSpace):
    #     cpp.dolfin_error("interpolation.py",
    #                      "compute interpolation",
    #                      "Illegal function space for interpolation, not a FunctionSpace (%s)" % str(v))

    # Compute interpolation
    Pv = function.Function(V)
    Pv.interpolate(v)

    return Pv


def compiled_interpolation(expr, V, target):

    target_el = V.ufl_element()

    if target_el.value_size() > 1:
        # For mixed elements fetch only one element
        # Saves computation for vector/tensor elements,
        # no need to evaluate at same points for each vector
        # component
        #
        # TODO: We can get unique subelements or unique
        #       points for evaluation
        sub_elements = target_el.sub_elements()

        # We can handle only all sub elements equal case
        assert all([sub_elements[0] == x for x in sub_elements])
        target_el = sub_elements[0]

    # Identify points at which to evaluate the expression
    fiat_element = ffc.fiatinterface.create_element(target_el)

    if not all(x == "affine" for x in fiat_element.mapping()):
        raise NotImplementedError("Only affine mapped function spaces supported")

    nodes = []
    for dual in fiat_element.dual_basis():
        pts, = dual.pt_dict.keys()
        nodes.append(pts)

    nodes = np.asarray(nodes)

    module = jit.ffc_jit((expr, nodes))
    kernel = module.tabulate_expression

    ffi = cffi.FFI()
    # Register complex types
    numba.cffi_support.register_type(ffi.typeof('double _Complex'),
                                     numba.types.complex128)
    numba.cffi_support.register_type(ffi.typeof('float _Complex'),
                                     numba.types.complex64)

    reference_geometry = np.asarray(fiat_element.ref_el.get_vertices())

    # Unpack mesh and dofmap data
    mesh = V.mesh
    c = mesh.topology.connectivity(2, 0).connections()
    pos = mesh.topology.connectivity(2, 0).pos()
    geom = mesh.geometry.points
    dofmap = V.dofmap.dof_array

    # Prepare coefficients and their dofmaps
    # Global vectors and dofmaps are prepared here, local are
    # fetched inside hot cell-loop
    #
    # FIXME: Is np.asarray on petsc4py vector safe?
    coeffs = ufl.algorithms.analysis.extract_coefficients(expr)
    coeffs_dofmaps = List.empty_list(numba.types.Array(numba.typeof(dofmap[0]), 1, "C", readonly=True))
    coeffs_vectors = List.empty_list(numba.types.Array(numba.typeof(PETSc.ScalarType()), 1, "C"))

    for coeff in coeffs:
        coeffs_dofmaps.append(coeff.function_space.dofmap.dof_array)
        coeffs_vectors.append(np.asarray(coeff.vector))

    local_coeffs_sizes = np.asarray([coeff.function_space.element.space_dimension() for coeff in coeffs], dtype=np.int)
    local_coeffs_size = np.sum(local_coeffs_sizes, dtype=np.int)

    # Prepare and pack constants
    constants = ufl.algorithms.analysis.extract_constants(expr)
    constants_vector = np.array([], dtype=PETSc.ScalarType())
    if len(constants) > 0:
        constants_vector = np.hstack([c.value.flatten() for c in constants])

    # Num DOFs of the target element
    local_b_size = V.element.space_dimension()
    num_coeffs = len(coeffs_vectors)

    @numba.njit
    def assemble_vector_ufc(b, kernel, topology, geometry, dofmap, coeffs_vectors, coeffs_dofmaps, const_vector):
        connections, pos = topology
        coordinate_dofs = np.zeros(reference_geometry.shape)
        coeffs = np.zeros(local_coeffs_size, dtype=PETSc.ScalarType)
        b_local = np.zeros(local_b_size, dtype=PETSc.ScalarType)

        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i + 1] - pos[i]
            c = connections[cell:cell + num_vertices]
            for j in range(reference_geometry.shape[0]):
                for k in range(reference_geometry.shape[1]):
                    coordinate_dofs[j, k] = geometry[c[j], k]
            b_local.fill(0.0)

            offset = 0
            for j in range(num_coeffs):
                local_dofsize = local_coeffs_sizes[j]
                for k in range(local_dofsize):
                    coeffs[offset + k] = coeffs_vectors[j][coeffs_dofmaps[j][i * local_dofsize + k]]
                offset += local_dofsize

            kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs),
                   ffi.from_buffer(const_vector), ffi.from_buffer(coordinate_dofs))

            for j in range(local_b_size):
                b[dofmap[i * local_b_size + j]] = b_local[j]

    with target.vector.localForm() as b:
        b.set(0.0)
        assemble_vector_ufc(np.asarray(b), kernel, (c, pos), geom,
                            dofmap, coeffs_vectors, coeffs_dofmaps, constants_vector)
