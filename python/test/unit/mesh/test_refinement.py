
import pytest
from dolfin import UnitSquareMesh, UnitCubeMesh, MPI
from dolfin.cpp.refinement import refine

def test_RefineUnitSquareMesh():
    """Refine mesh of unit square."""
    mesh = UnitSquareMesh(MPI.comm_world, 5, 7)
    mesh = refine(mesh, False)
    assert mesh.num_entities_global(0) == 165
    assert mesh.num_entities_global(2) == 280


def test_RefineUnitCubeMesh():
    """Refine mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.comm_world, 5, 7, 9)
    mesh = refine(mesh, False)
    assert mesh.num_entities_global(0) == 3135
    assert mesh.num_entities_global(3) == 15120
