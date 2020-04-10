"""Unit tests for the DiscreteOperator class"""

# Copyright (C) 2015 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# from math import sqrt

# import pytest
# from petsc4py import PETSc

# from dolfinx FunctionSpace, UnitCubeMesh, UnitSquareMesh
# from dolfinx.cpp.fem import DiscreteOperators
# from dolfinx_utils.test.skips import skip_in_parallel


# @skip_in_parallel
# def test_gradient():
#     """Test discrete gradient computation (typically used for curl-curl
#     AMG preconditioners"""

#     def compute_discrete_gradient(mesh):
#         V = FunctionSpace(mesh, ("Lagrange", 1))
#         W = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", 1))
#         G = DiscreteOperators.build_gradient(W._cpp_object, V._cpp_object)
#         assert G.getRefCount() == 1
#         num_edges = mesh.topology.index_map(1).size_global
#         m, n = G.getSize()
#         assert m == num_edges
#         assert n == mesh.topology.index_map(0).size_global
#         assert round(
#             G.norm(PETSc.NormType.FROBENIUS) - sqrt(2.0 * num_edges),
#             8) == 0.0

#     meshes = [
#         UnitSquareMesh(MPI.COMM_WORLD, 11, 6),
#         UnitCubeMesh(MPI.COMM_WORLD, 4, 3, 7)
#     ]
#     for mesh in meshes:
#         compute_discrete_gradient(mesh)


# def test_incompatible_spaces():
#     "Test that error is thrown when function spaces are not compatible"

#     mesh = UnitSquareMesh(MPI.COMM_WORLD, 13, 7)
#     V = FunctionSpace(mesh, ("Lagrange", 1))
#     W = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", 1))
#     with pytest.raises(RuntimeError):
#         DiscreteOperators.build_gradient(V._cpp_object, W._cpp_object)
#     with pytest.raises(RuntimeError):
#         DiscreteOperators.build_gradient(V._cpp_object, V._cpp_object)
#     with pytest.raises(RuntimeError):
#         DiscreteOperators.build_gradient(W._cpp_object, W._cpp_object)

#     V = FunctionSpace(mesh, ("Lagrange", 2))
#     with pytest.raises(RuntimeError):
#         DiscreteOperators.build_gradient(W._cpp_object, V._cpp_object)
