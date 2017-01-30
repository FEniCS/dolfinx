from dolfin import *
import pytest
from dolfin_utils.test import skip_in_parallel

# test case with interface-edge overlap
@pytest.fixture
def test_case_1(M, N):
    multimesh = MultiMesh()
    mesh0 = UnitSquareMesh(M, M)
    mesh1 = RectangleMesh(Point(0.25, 0.25), Point(0.75, 0.75), N, N)
    multimesh.add(mesh0)
    multimesh.add(mesh1)
    multimesh.build()
    return multimesh

# test case with squares on the diagonal
@pytest.fixture
def test_case_2(width, offset, Nx):

    # Mesh width (must be less than 1)
    assert width < 1

    # Mesh placement (must be less than the width)
    assert offset < width

    # Background mesh
    mesh_0 = UnitSquareMesh(Nx, Nx)

    # Create multimesh
    multimesh = MultiMesh()
    multimesh.add(mesh_0)

    # Now we have num_parts = 1
    num_parts = multimesh.num_parts()

    while num_parts*offset + width < 1:
        a = num_parts*offset
        b = a + width
        mesh_top = RectangleMesh(Point(a,a), Point(b,b), Nx, Nx)
        multimesh.add(mesh_top)
        num_parts = multimesh.num_parts()

    multimesh.build()
    return multimesh


test_cases = [test_case_1(4,3), 
              test_case_2(DOLFIN_PI/5, 0.1111, 3)]

@skip_in_parallel
@pytest.mark.parametrize("multimesh", test_cases)
def test_cut_cell_has_quadrature(multimesh):
    # Test that every cut cell has a nontrivial interface quadrature rule
    for part in range(multimesh.num_parts()):
        for cell in multimesh.cut_cells(part):
            assert multimesh.quadrature_rule_interface(part, cell)


@skip_in_parallel
@pytest.mark.parametrize("multimesh", test_cases)
def test_multimesh_cell_types(multimesh):
    # Test that every cell in the multimesh is either cut, uncut, or covered
    for part in range(multimesh.num_parts()):
        cells = set(range(multimesh.part(part).num_cells()))
        cut_cells = set(multimesh.cut_cells(part))
        uncut_cells = set(multimesh.uncut_cells(part))
        covered_cells = set(multimesh.covered_cells(part))

        assert cut_cells.union(uncut_cells).union(covered_cells) == cells

        assert cut_cells.intersection(uncut_cells) == set()
        assert cut_cells.intersection(covered_cells) == set()
        assert uncut_cells.intersection(covered_cells) == set()

