"""Unit tests for graph building"""

# Copyright (C) 2013 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from dolfin import *


def test_build_from_mesh_simple():
    """Build mesh graph """

    mesh = UnitCubeMesh(MPI.comm_world, 16, 16, 16)
    D = mesh.topology.dim
    GraphBuilder.local_graph(mesh, D, 0)
    GraphBuilder.local_graph(mesh, D, 1)
    GraphBuilder.local_graph(mesh, 2, D)
    GraphBuilder.local_graph(mesh, 1, D)
    GraphBuilder.local_graph(mesh, 0, D)
