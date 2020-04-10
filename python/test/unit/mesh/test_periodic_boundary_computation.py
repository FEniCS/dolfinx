# # Copyright (C) 2013 Mikael Mortensen
# #
# # This file is part of DOLFINX (https://www.fenicsproject.org)
# #
# # SPDX-License-Identifier:    LGPL-3.0-or-later

# import numpy as np

# from dolfinx import UnitSquareMesh
# from dolfinx.cpp.mesh import PeriodicBoundaryComputation
# from dolfinx_utils.test.skips import skip_in_parallel


# def periodic_boundary(x):
#     return np.isclose(x[:, 0], 0.0)


# @pytest.fixture
# def mesh():
#     return UnitSquareMesh(MPI.COMM_WORLD, 4, 4)


# @skip_in_parallel
# def test_ComputePeriodicPairs(mesh):
#     # Verify that correct number of periodic pairs are computed
#     vertices = PeriodicBoundaryComputation.compute_periodic_pairs(
#         mesh, periodic_boundary, 0, np.finfo(float).eps)
#     edges = PeriodicBoundaryComputation.compute_periodic_pairs(
#         mesh, periodic_boundary, 1, np.finfo(float).eps)
#     assert len(vertices) == 5
#     assert len(edges) == 4


# @skip_in_parallel
# def test_MastersSlaves(mesh):
#     # Verify that correct number of masters and slaves are marked
#     mf = PeriodicBoundaryComputation.masters_slaves(mesh, periodic_boundary, 0, np.finfo(float).eps)
#     assert len(np.where(mf.array() == 1)[0]) == 5
#     assert len(np.where(mf.array() == 2)[0]) == 5

#     mf = PeriodicBoundaryComputation.masters_slaves(mesh, periodic_boundary, 1, np.finfo(float).eps)
#     assert len(np.where(mf.array() == 1)[0]) == 4
#     assert len(np.where(mf.array() == 2)[0]) == 4
