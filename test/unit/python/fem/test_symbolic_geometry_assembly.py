#!/usr/bin/env py.test
from dolfin import *
import ufl
import numpy
import sys
import pytest
from dolfin_utils.test import set_parameters_fixture, skip_in_parallel, fixture

from ufl.classes import CellOrientation, CellNormal, CellCoordinate, \
CellOrigin, Jacobian, JacobianInverse, JacobianDeterminant

# This was for debugging, don't enable this permanently here in tests
# parameters["reorder_dofs_serial"] = False
any_representation = \
  set_parameters_fixture("form_compiler.representation",
                             ["quadrature", "uflacs"])
uflacs_representation_only \
    = set_parameters_fixture("form_compiler.representation",
                             ["uflacs"])


def create_mesh(vertices, cells, cellname="simplex"):
    """Given list of vertex coordinate tuples and cell vertex index
tuples, build and return a mesh.

    """

    # Get dimensions
    gdim = len(vertices[0])

    # Automatic choice of cellname for simplices
    if cellname == "simplex":
        num_cell_vertices = len(cells[0])
        cellname = {1: "vertex",
                    2: "interval",
                    3: "triangle",
                    4: "tetrahedron",
                    }[num_cell_vertices]

    # Indirect error checking and determination of tdim via ufl
    ufl_cell = ufl.Cell(cellname, gdim)
    tdim = ufl_cell.topological_dimension()

    # Create mesh to return
    mesh = Mesh()

    # Open mesh in editor
    me = MeshEditor()
    me.open(mesh, cellname, tdim, gdim)

    # Add vertices to mesh
    me.init_vertices(len(vertices))
    for i, v in enumerate(vertices):
        me.add_vertex(i, numpy.array(v, dtype='float'))

    # Add cells to mesh
    me.init_cells(len(cells))
    for i, c in enumerate(cells):
        me.add_cell(i, numpy.array(c, dtype='uint'))

    me.close()

    return mesh


def create_line_mesh(vertices):
    """Given list of vertex coordinate tuples, build and return a mesh of
intervals.

    """

    # Get dimensions
    gdim = len(vertices[0])
    tdim = 1

    # Automatic choice of cellname for simplices
    cellname = "interval"

    # Indirect error checking and determination of tdim via ufl
    ufl_cell = ufl.Cell(cellname, gdim)
    assert tdim == ufl_cell.topological_dimension()

    # Create mesh to return
    mesh = Mesh()

    # Open mesh in editor
    me = MeshEditor()
    me.open(mesh, cellname, tdim, gdim)

    # Add vertices to mesh
    nv = len(vertices)
    me.init_vertices(nv)

    for i, v in enumerate(vertices):
        me.add_vertex(i, numpy.array(v, dtype='float'))

    # TODO: Systematically swap around vertex ordering to test cell orientation

    # Add cells to mesh
    me.init_cells(nv-1)
    for i in range(nv-1):
        c = (i, i+1)
        me.add_cell(i, numpy.array(c, dtype='uint'))

    me.close()

    return mesh

line_resolution = 8


@fixture
def line1d(request):
    n = line_resolution
    us = [i/float(n-1) for i in range(n)]
    vertices = [(u**2,) for u in us]
    return create_line_mesh(vertices)


@fixture
def rline1d(request):
    n = line_resolution
    us = [i/float(n-1) for i in range(n)]
    vertices = [(u**2,) for u in us]
    vertices = list(reversed(vertices))  # same as line1d, just reversed here
    return create_line_mesh(vertices)


@fixture
def line2d(request):
    n = line_resolution
    us = [i/float(n-1) for i in range(n)]
    vertices = [(cos(DOLFIN_PI*u), sin(DOLFIN_PI*u)) for u in us]
    mesh = create_line_mesh(vertices)
    mesh.init_cell_orientations(Expression(("0.0", "1.0"), degree=0))
    return mesh


@fixture
def rline2d(request):
    n = line_resolution
    us = [i/float(n-1) for i in range(n)]
    vertices = [(cos(DOLFIN_PI*u), sin(DOLFIN_PI*u)) for u in us]
    vertices = list(reversed(vertices))  # same as line2d, just reversed here
    mesh = create_line_mesh(vertices)
    mesh.init_cell_orientations(Expression(("0.0", "1.0"), degree=0))
    return mesh


@fixture
def line3d(request):
    n = line_resolution
    us = [i/float(n-1) for i in range(n)]
    vertices = [(cos(4.0*DOLFIN_PI*u),
                 sin(4.0*DOLFIN_PI*u),
                 2.0*u) for u in us]
    mesh = create_line_mesh(vertices)
    return mesh


@fixture
def rline3d(request):
    n = line_resolution
    us = [i/float(n-1) for i in range(n)]
    vertices = [(cos(4.0*DOLFIN_PI*u),
                 sin(4.0*DOLFIN_PI*u),
                 2.0*u) for u in us]
    vertices = list(reversed(vertices))  # same as line3d, just reversed here
    mesh = create_line_mesh(vertices)
    return mesh


@fixture
def square2d(request):
    cellname = "triangle"
    side = sqrt(sqrt(3.0))
    vertices = [
        (0.0, 0.0),
        (side, side),
        (side, 0.0),
        (0.0, side),
        ]
    cells = [
        (0, 1, 2),
        (0, 1, 3),
        ]
    mesh = create_mesh(vertices, cells)
    return mesh


@fixture
def square3d(request):
    cellname = "triangle"
    vertices = [
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        ]
    cells = [
        (0, 1, 2),
        (0, 1, 3),
        ]
    mesh = create_mesh(vertices, cells)
    mesh.init_cell_orientations(Expression(("0.0", "0.0", "1.0"), degree=0))

    return mesh


@skip_in_parallel
def test_line_meshes(line1d, line2d, line3d, rline1d, rline2d, rline3d):
    "Check some properties of the meshes created for these tests."
    assert line1d.geometry().dim() == 1
    assert line2d.geometry().dim() == 2
    assert line3d.geometry().dim() == 3
    assert line1d.topology().dim() == 1
    assert line2d.topology().dim() == 1
    assert line3d.topology().dim() == 1


@skip_in_parallel
def test_write_line_meshes_to_files(line1d, line2d, line3d, rline1d, rline2d,
                                    rline3d, uflacs_representation_only):
    # Enable to write meshes to file for inspection (plot doesn't work
    # for 1d in 2d/3d)
    # CellNormal is only supported by uflacs
    if 0:
        File("line1d.xdmf") << line1d
        File("line2d.xdmf") << line2d
        File("line3d.xdmf") << line3d
        File("rline1d.xdmf") << rline1d
        File("rline2d.xdmf") << rline2d
        File("rline3d.xdmf") << rline3d
        File("line2dnormal.xdmf") << project(CellNormal(line2d),
                                             VectorFunctionSpace(line2d,
                                                                 "DG", 0))
        File("rline2dnormal.xdmf") << project(CellNormal(rline2d),
                                              VectorFunctionSpace(rline2d,
                                                                  "DG", 0))


@skip_in_parallel
@pytest.mark.parametrize("mesh", [
    line1d(None),
    line2d(None),
    line3d(None),
    rline1d(None),
    rline2d(None),
    rline3d(None), ])
def test_manifold_line_geometry(mesh, uflacs_representation_only):
    assert uflacs_representation_only == "uflacs"
    assert parameters["form_compiler"]["representation"] == "uflacs"

    gdim = mesh.geometry().dim()
    tdim = mesh.topology().dim()

    # Create cell markers and integration measure
    mf = CellFunction("size_t", mesh)
    dx = Measure("dx", domain=mesh, subdomain_data=mf)

    # Create symbolic geometry for current mesh
    x = SpatialCoordinate(mesh)
    X = CellCoordinate(mesh)
    co = CellOrientation(mesh)
    cn = CellNormal(mesh)
    J = Jacobian(mesh)
    detJ = JacobianDeterminant(mesh)
    K = JacobianInverse(mesh)
    vol = CellVolume(mesh)
    h = CellDiameter(mesh)
    R = Circumradius(mesh)

    # Check that length computed via integral doesn't change with
    # refinement
    length = assemble(1.0*dx)
    mesh2 = refine(mesh)
    assert mesh2.num_cells() == 2*mesh.num_cells()
    dx2 = Measure("dx")
    length2 = assemble(1.0*dx2(mesh2))
    assert round(length - length2, 7) == 0.0

    # Check that number of cells can be computed correctly by scaling
    # integral by |detJ|^-1
    num_cells = assemble(1.0/abs(detJ)*dx)
    assert round(num_cells - mesh.num_cells(), 7) == 0.0

    # Check that norm of Jacobian column matches detJ and volume
    assert round(length - assemble(sqrt(J[:, 0]**2)/abs(detJ)*dx), 7) == 0.0
    assert round(assemble((vol-abs(detJ))*dx), 7) == 0.0
    assert round(length - assemble(vol/abs(detJ)*dx), 7) == 0.0

    coords = mesh.coordinates()
    cells = mesh.cells()

    # Checks on each cell separately
    for k in range(mesh.num_cells()):
        # Mark current cell
        mf.set_all(0)
        mf[k] = 1

        x0 = Constant(tuple(coords[cells[k][0], :]))

        # Integrate x components over a cell and compare with midpoint
        # computed from coords
        for j in range(gdim):
            xm = 0.5*(coords[cells[k][0], j] + coords[cells[k][1], j])
            assert round(assemble(x[j]/abs(detJ)*dx(1)) - xm, 7) == 0.0

        # Jacobian column is pointing away from x0
        assert assemble(dot(J[:, 0], x-x0)*dx(1)) > 0.0

        # Check affine coordinate relations x=x0+J*X, X=K*(x-x0), K*J=I
        assert round(assemble((x - (x0+J*X))**2*dx(1)), 7) == 0.0
        assert round(assemble((X - K*(x-x0))**2*dx(1)), 7) == 0.0
        assert round(assemble((K*J - Identity(tdim))**2*dx(1)), 7) == 0.0

        # Check cell diameter and circumradius
        assert round(assemble(h/vol*dx(1)) - Cell(mesh, k).h(), 7) == 0.0
        assert round(assemble(R/vol*dx(1)) - Cell(mesh, k).circumradius(), 7) == 0.0

        # Jacobian column is orthogonal to cell normal
        if gdim == 2:
            assert round(assemble(dot(J[:, 0], cn)*dx(1)), 7) == 0.0

            # Create 3d tangent and cell normal vectors
            tangent = as_vector((J[0, 0], J[1, 0], 0.0))
            tangent = co * tangent / sqrt(tangent**2)
            normal = as_vector((cn[0], cn[1], 0.0))
            up = cross(tangent, normal)

            # Check that t,n,up are orthogonal
            assert round(assemble(dot(tangent, normal)*dx(1)), 7) == 0.0
            assert round(assemble(dot(tangent, up)*dx(1)), 7) == 0.0
            assert round(assemble(dot(normal, up)*dx(1)), 7) == 0.0
            assert round(assemble((cross(up, tangent) - normal)**2*dx(1)),
                         7) == 0.0

            assert round(assemble(up**2*dx(1)), 7) > 0.0
            assert round(assemble((up[0]**2 + up[1]**2)*dx(1)), 7) == 0.0
            assert round(assemble(up[2]*dx(1)), 7) > 0.0


@skip_in_parallel
def test_manifold_area(square3d, any_representation):
    """Integrate literal expressions over manifold cells, no function
spaces involved."""
    mesh = square3d
    area = sqrt(3.0)  # known area of mesh

    # Assembling mesh area scaled by a literal
    assert round(assemble(0.0*dx(mesh)) - 0.0*area, 7) == 0.0
    assert round(assemble(1.0*dx(mesh)) - 1.0*area, 7) == 0.0
    assert round(assemble(3.0*dx(mesh)) - 3.0*area, 7) == 0.0


@skip_in_parallel
def test_manifold_dg0_functions(square3d, any_representation):
    mesh = square3d
    area = sqrt(3.0)  # known area of mesh

    mf = CellFunction("size_t", mesh)
    mf[0] = 0
    mf[1] = 1
    dx = Measure("dx", domain=mesh, subdomain_data=mf)

    x = SpatialCoordinate(mesh)

    U0 = FunctionSpace(mesh, "DG", 0)
    V0 = VectorFunctionSpace(mesh, "DG", 0)

    # Project constants to scalar and vector DG0 spaces on manifold
    u0 = project(1.0, U0)
    v0v = (1.0, 2.0, 3.0)
    v0 = project(as_vector(v0v), V0)
    assert round(sum(u0.vector().array()) - 2*1, 7) == 0.0
    assert round(sum(v0.vector().array()) - 2*(1+2+3), 7) == 0.0

    # Integrate piecewise constant functions over manifold cells
    assert round(assemble(u0*dx(0)) - 0.5*area) == 0.0
    assert round(assemble(u0*dx(1)) - 0.5*area) == 0.0
    assert round(assemble(u0*dx) - area) == 0.0
    assert round(assemble(v0[0]*dx) - v0v[0]*area) == 0.0
    assert round(assemble(v0[1]*dx) - v0v[1]*area) == 0.0
    assert round(assemble(v0[2]*dx) - v0v[2]*area) == 0.0

    # Project x to scalar and vector DG0 spaces on manifold
    u0x = project(x[0], U0)  # cell averages of x[0]: 2/3, 1/3, sum = 3/3
    v0x = project(x, V0)  # cell averages of x[:]: (2/3, 1/3, 2/3), (1/3, 2/3, 2/3), sum = 10/3
    assert round(sum(u0x.vector().array()) - 3.0/3.0, 7) == 0.0
    assert round(sum(v0x.vector().array()) - 10.0/3.0, 7) == 0.0

    # Evaluate in all corners and cell midpoints, value should be the
    # same constant everywhere
    points = [
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0/3.0, 2.0/3.0, 2.0/3.0),
        (2.0/3.0, 1.0/3.0, 2.0/3.0),
        ]
    for point in points:
        assert round(sum((v0(point) - numpy.asarray(v0v))**2), 7) == 0.0


@skip_in_parallel
def test_manifold_cg1_functions(square3d, any_representation):
    mesh = square3d
    area = sqrt(3.0)  # known area of mesh

    mf = CellFunction("size_t", mesh)
    mf[0] = 0
    mf[1] = 1
    dx = Measure("dx", domain=mesh, subdomain_data=mf)

    # We need unit testing of some symbolic quantities to pinpoint any
    # sign problems etc. in the right places
    x = SpatialCoordinate(mesh)

    U1 = FunctionSpace(mesh, "CG", 1)
    V1 = VectorFunctionSpace(mesh, "CG", 1)

    # Project piecewise linears to scalar and vector CG1 spaces on
    # manifold
    u1 = project(x[0], U1)
    v1 = project(x, V1)
    # exact x in vertices is [0,0,0, 1,1,1, 1,0,0, 0,1,0],
    # so sum(x[0] for each vertex) is therefore sum(0 1 1 0):
    assert round(sum(u1.vector().array()) - (0+1+1+0), 7) == 0.0
    # and sum(x components for each vertex) is sum(1, 3, 1, 1):
    assert round(sum(v1.vector().array()) - (1+3+1+1), 7) == 0.0

    # Integrate piecewise constant functions over manifold cells,
    # computing midpoint coordinates
    midpoints = [
        (1.0/3.0, 2.0/3.0, 2.0/3.0),
        (2.0/3.0, 1.0/3.0, 2.0/3.0),
        ]
    mp = midpoints
    assert round(assemble(u1*dx(0)) - mp[0][0]) == 0.0
    assert round(assemble(u1*dx(1)) - mp[1][0]) == 0.0
    assert round(assemble(v1[0]*dx(0)) - mp[0][0]) == 0.0
    assert round(assemble(v1[1]*dx(0)) - mp[0][1]) == 0.0
    assert round(assemble(v1[2]*dx(0)) - mp[0][2]) == 0.0
    assert round(assemble(v1[0]*dx(1)) - mp[1][0]) == 0.0
    assert round(assemble(v1[1]*dx(1)) - mp[1][1]) == 0.0
    assert round(assemble(v1[2]*dx(1)) - mp[1][2]) == 0.0


@skip_in_parallel
def test_manifold_coordinate_projection(square3d, any_representation):
    mesh = square3d

    # Project x to a CG1 Function, i.e. setting up for v1(x) = x
    V1 = VectorFunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)
    v1 = project(x, V1)

    # Check that v1(x) = x holds component-wise in squared l2 norm
    assert round(assemble((v1-x)**2*dx), 7) == 0.0
    assert round(assemble((v1[0]-x[0])**2*dx), 7) == 0.0
    assert round(assemble((v1[1]-x[1])**2*dx), 7) == 0.0
    assert round(assemble((v1[2]-x[2])**2*dx), 7) == 0.0
    assert round(assemble((v1[0]-x[0])**2*dx(0)), 7) == 0.0
    assert round(assemble((v1[1]-x[1])**2*dx(0)), 7) == 0.0
    assert round(assemble((v1[2]-x[2])**2*dx(0)), 7) == 0.0
    assert round(assemble((v1[0]-x[0])**2*dx(1)), 7) == 0.0
    assert round(assemble((v1[1]-x[1])**2*dx(1)), 7) == 0.0
    assert round(assemble((v1[2]-x[2])**2*dx(1)), 7) == 0.0


@skip_in_parallel
def test_manifold_point_evaluation(square3d, any_representation):
    mesh = square3d

    # Project x to a CG1 Function, i.e. setting up for v1(x) = x
    V1 = VectorFunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)
    v1 = project(x, V1)

    # Evaluate in all corners and cell midpoints,
    # value should equal the evaluation coordinate
    # Because bounding box tree doesn't handle manifolds,
    # we have to specify which cell each point is in.
    points = [[
        (0.0, 0.0, 1.0),  # vertex of both cells
        (1.0, 1.0, 1.0),  # vertex of both cells
        (1.0, 0.0, 0.0),  # vertex of cell 0 only
        (2.0/3.0, 1.0/3.0, 2.0/3.0),  # midpoint of cell 0
        ], [
        (0.0, 0.0, 1.0),  # vertex of both cells
        (1.0, 1.0, 1.0),  # vertex of both cells
        (0.0, 1.0, 0.0),  # vertex of cell 1 only
        (1.0/3.0, 2.0/3.0, 2.0/3.0),  # midpoint of cell 1
        ]]
    values = numpy.zeros(3)
    bb = mesh.bounding_box_tree()
    for cellid in (0, 1):
        for point in points[cellid]:
            v1.eval_cell(values, numpy.asarray(point), Cell(mesh, cellid))
            assert round(values[0] - point[0], 7) == 0.0  # expecting v1(x) = x
            assert round(values[1] - point[1], 7) == 0.0  # expecting v1(x) = x
            assert round(values[2] - point[2], 7) == 0.0  # expecting v1(x) = x
            # print cellid, [round(v,2) for v in point],
            # [round(v,2) for v in values]


# Some symbolic quantities are only available through uflacs
@skip_in_parallel
def test_manifold_symbolic_geometry(square3d, uflacs_representation_only):
    mesh = square3d
    assert mesh.num_cells() == 2
    gdim = mesh.geometry().dim()
    tdim = mesh.topology().dim()

    area = sqrt(3.0)  # known area of mesh
    A = area/2.0  # area of single cell
    Aref = 0.5  # 0.5 is the area of the UFC reference triangle

    mf = CellFunction("size_t", mesh)
    mf[0] = 0
    mf[1] = 1
    dx = Measure("dx", domain=mesh, subdomain_data=mf)

    U0 = FunctionSpace(mesh, "DG", 0)
    V0 = VectorFunctionSpace(mesh, "DG", 0)

    # 0 means up=+1.0, 1 means down=-1.0
    orientations = mesh.cell_orientations()
    assert orientations[0] == 1  # down
    assert orientations[1] == 0  # up

    # Check cell orientation, should be -1.0 (down) and +1.0 (up) on
    # the two cells respectively by construction
    co = CellOrientation(mesh)
    co0 = assemble(co/A*dx(0))
    co1 = assemble(co/A*dx(1))
    assert round(abs(co0) - 1.0, 7) == 0.0  # should be +1 or -1
    assert round(abs(co1) - 1.0, 7) == 0.0  # should be +1 or -1
    assert round(co1 + co0, 7) == 0.0  # should cancel out
    assert round(co0 - -1.0, 7) == 0.0  # down
    assert round(co1 - +1.0, 7) == 0.0  # up

    # Check cell normal directions component for component
    cn = CellNormal(mesh)
    assert assemble(cn[0]/A*dx(0)) > 0.0
    assert assemble(cn[0]/A*dx(1)) < 0.0
    assert assemble(cn[1]/A*dx(0)) < 0.0
    assert assemble(cn[1]/A*dx(1)) > 0.0
    assert assemble(cn[2]/A*dx(0)) > 0.0
    assert assemble(cn[2]/A*dx(1)) > 0.0
    # Check cell normal normalization
    assert round(assemble(cn**2/A*dx(0)) - 1.0, 7) == 0.0
    assert round(assemble(cn**2/A*dx(1)) - 1.0, 7) == 0.0

    # Check coordinates with various consistency checking
    x = SpatialCoordinate(mesh)
    X = CellCoordinate(mesh)
    J = Jacobian(mesh)
    detJ = JacobianDeterminant(mesh)  # pseudo-determinant
    K = JacobianInverse(mesh)  # pseudo-inverse
    vol = CellVolume(mesh)
    h = CellDiameter(mesh)
    R = Circumradius(mesh)

    # This is not currently implemented in uflacs:
    # x0 = CellOrigin(mesh)
    # But by happy accident, x0 is the same vertex for both our triangles:
    x0 = as_vector((0.0, 0.0, 1.0))

    # Checks on each cell separately
    for k in range(mesh.num_cells()):
        # Mark current cell
        mf.set_all(0)
        mf[k] = 1

        # Check integration area vs detJ
        # Validate known cell area A
        assert round(assemble(1.0*dx(1)) - A, 7) == 0.0
        assert round(assemble(1.0/A*dx(1)) - 1.0, 7) == 0.0
        assert round(assemble(A*dx(1)) - A**2, 7) == 0.0
        # Compare abs(detJ) to A
        A2 = Aref*abs(detJ)
        assert round(assemble((A-A2)**2*dx(1)) - 0.0, 7) == 0.0
        assert round(assemble(1.0/A2*dx(1)) - 1.0, 7) == 0.0
        assert round(assemble(A2*dx(1)) - A**2, 7) == 0.0
        # Validate cell orientation
        assert round(assemble(co*dx(1)) - A*(1 if k == 1 else -1), 7) == 0.0
        # Compare co*detJ to A (detJ is pseudo-determinant with sign
        # restored, *co again is equivalent to abs())
        A3 = Aref*co*detJ
        assert round(assemble((A-A3)**2*dx(1)) - 0.0, 7) == 0.0
        assert round(assemble((A2-A3)**2*dx(1)) - 0.0, 7) == 0.0
        assert round(assemble(1.0/A3*dx(1)) - 1.0, 7) == 0.0
        assert round(assemble(A3*dx(1)) - A**2, 7) == 0.0
        # Compare vol to A
        A4 = vol
        assert round(assemble((A-A4)**2*dx(1)) - 0.0, 7) == 0.0
        assert round(assemble((A2-A4)**2*dx(1)) - 0.0, 7) == 0.0
        assert round(assemble((A3-A4)**2*dx(1)) - 0.0, 7) == 0.0
        assert round(assemble(1.0/A4*dx(1)) - 1.0, 7) == 0.0
        assert round(assemble(A4*dx(1)) - A**2, 7) == 0.0

        # Check cell diameter and circumradius
        assert round(assemble(h/vol*dx(1)) - Cell(mesh, k).h(), 7) == 0.0
        assert round(assemble(R/vol*dx(1)) - Cell(mesh, k).circumradius(), 7) == 0.0

        # Check integral of reference coordinate components over reference
        # triangle: \int_0^1 \int_0^{1-x} x dy dx = 1/6
        Xmp = (1.0/6.0, 1.0/6.0)
        for j in range(tdim):
            # Scale by detJ^-1 to get reference cell integral
            assert round(assemble(X[j]/abs(detJ)*dx(1)) - Xmp[j], 7) == 0.0

        # Check average of physical coordinate components over each cell:
        xmp = [(2.0/3.0, 1.0/3.0, 2.0/3.0),  # midpoint of cell 0
               (1.0/3.0, 2.0/3.0, 2.0/3.0),  # midpoint of cell 1
               ]
        for i in range(gdim):
            # Scale by A^-1 to get average of x, not integral
            assert round(assemble(x[i]/A*dx(1)) - xmp[k][i], 7) == 0.0

    # Check affine coordinate relations x=x0+J*X, X=K*(x-x0), K*J=I
    assert round(assemble((x - (x0+J*X))**2*dx), 7) == 0.0
    assert round(assemble((X - K*(x-x0))**2*dx), 7) == 0.0
    assert round(assemble((K*J - Identity(2))**2/A*dx), 7) == 0.0


@skip_in_parallel
def test_manifold_piola_mapped_functions(square3d, any_representation):
    mesh = square3d
    area = sqrt(3.0)  # known area of mesh
    A = area/2.0

    mf = CellFunction("size_t", mesh)
    mf[0] = 0
    mf[1] = 1
    dx = Measure("dx", domain=mesh, subdomain_data=mf)

    x = SpatialCoordinate(mesh)

    J = Jacobian(mesh)
    detJ = JacobianDeterminant(mesh)  # pseudo-determinant
    K = JacobianInverse(mesh)  # pseudo-inverse

    Q1 = VectorFunctionSpace(mesh, "CG", 1)
    U1 = VectorFunctionSpace(mesh, "DG", 1)
    V1 = FunctionSpace(mesh, "N1div", 1)
    W1 = FunctionSpace(mesh, "N1curl", 1)

    dq = TestFunction(Q1)
    du = TestFunction(U1)
    dv = TestFunction(V1)
    dw = TestFunction(W1)

    assert U1.ufl_element().mapping() == "identity"
    assert V1.ufl_element().mapping() == "contravariant Piola"
    assert W1.ufl_element().mapping() == "covariant Piola"

    if any_representation != "uflacs":
        return

    # Check that projection test fails if it should fail:
    vec = Constant((0.0, 0.0, 0.0))
    q1 = project(vec, Q1)
    u1 = project(vec, U1)
    v1 = project(vec, V1)
    w1 = project(vec, W1)
    # Projection of zero gets us zero for all spaces
    assert assemble(q1**2*dx) == 0.0
    assert assemble(u1**2*dx) == 0.0
    assert assemble(v1**2*dx) == 0.0
    assert assemble(w1**2*dx) == 0.0
    # Changing vec to nonzero, check that dM/df != 0 at f=0
    vec = Constant((2.0, 2.0, 2.0))
    assert round(assemble(derivative((q1-vec)**2*dx, q1)).norm('l2') -
                 assemble(-4.0*sum(dq)*dx).norm('l2'), 7) == 0.0
    assert round(assemble(derivative((u1-vec)**2*dx, u1)).norm('l2') -
                 assemble(-4.0*sum(du)*dx).norm('l2'), 7) == 0.0
    assert round(assemble(derivative((v1-vec)**2*dx, v1)).norm('l2') -
                 assemble(-4.0*sum(dv)*dx).norm('l2'), 7) == 0.0
    assert round(assemble(derivative((w1-vec)**2*dx, w1)).norm('l2') -
                 assemble(-4.0*sum(dw)*dx).norm('l2'), 7) == 0.0

    # Project piecewise linears to scalar and vector CG1 spaces on
    # manifold
    vec = Constant((1.0, 1.0, 1.0))
    q1 = project(vec, Q1)
    u1 = project(vec, U1)
    v1 = project(vec, V1)
    w1 = project(vec, W1)

    # If vec can be represented exactly in space this should be zero:
    assert round(assemble((q1-vec)**2*dx), 7) == 0.0
    assert round(assemble((u1-vec)**2*dx), 7) == 0.0
    assert round(assemble((v1-vec)**2*dx), 7) > 0.0  # Exact representation not possible?
    assert round(assemble((w1-vec)**2*dx), 7) > 0.0  # Exact representation not possible?

    # In the l2norm projection is correct these should be zero:
    assert round(assemble(derivative((q1-vec)**2*dx, v1)).norm('l2'), 7) == 0.0
    assert round(assemble(derivative((u1-vec)**2*dx, w1)).norm('l2'), 7) == 0.0
    assert round(assemble(derivative((v1-vec)**2*dx, v1)).norm('l2'), 7) == 0.0
    assert round(assemble(derivative((w1-vec)**2*dx, w1)).norm('l2'), 7) == 0.0

    # Hdiv mapping of a local constant vector should be representable
    # in hdiv conforming space
    vec = (1.0/detJ)*J*as_vector((3.0, 5.0))
    q1 = project(vec, Q1)
    u1 = project(vec, U1)
    v1 = project(vec, V1)
    w1 = project(vec, W1)
    assert round(assemble((q1-vec)**2*dx), 7) > 0.0  # Exact representation not possible?
    assert round(assemble((u1-vec)**2*dx), 7) == 0.0
    assert round(assemble((v1-vec)**2*dx), 7) == 0.0
    assert round(assemble((w1-vec)**2*dx), 7) > 0.0  # Exact representation not possible?

    # Hcurl mapping of a local constant vector should be representable
    # in hcurl conforming space
    vec = K.T*as_vector((5.0, 2.0))
    q1 = project(vec, Q1)
    u1 = project(vec, U1)
    v1 = project(vec, V1)
    w1 = project(vec, W1)
    assert round(assemble((q1-vec)**2*dx), 7) > 0.0  # Exact representation not possible?
    assert round(assemble((u1-vec)**2*dx), 7) == 0.0
    assert round(assemble((v1-vec)**2*dx), 7) > 0.0  # Exact representation not possible?
    assert round(assemble((w1-vec)**2*dx), 7) == 0.0


# Some symbolic quantities are only available through uflacs
@skip_in_parallel
def test_tetrahedron_symbolic_geometry(uflacs_representation_only):
    mesh = UnitCubeMesh(1, 1, 1)
    assert mesh.num_cells() == 6
    gdim = mesh.geometry().dim()
    tdim = mesh.topology().dim()

    area = 1.0  # known volume of mesh
    A = area/6.0  # volume of single cell
    Aref = 1.0/6.0  # the volume of the UFC reference tetrahedron

    mf = CellFunction("size_t", mesh)
    dx = Measure("dx", domain=mesh, subdomain_data=mf)

    U0 = FunctionSpace(mesh, "DG", 0)
    V0 = VectorFunctionSpace(mesh, "DG", 0)
    U1 = FunctionSpace(mesh, "DG", 1)
    V1 = VectorFunctionSpace(mesh, "DG", 1)

    # Check coordinates with various consistency checking
    x = SpatialCoordinate(mesh)
    X = CellCoordinate(mesh)
    J = Jacobian(mesh)
    detJ = JacobianDeterminant(mesh)
    K = JacobianInverse(mesh)
    vol = CellVolume(mesh)
    h = CellDiameter(mesh)
    R = Circumradius(mesh)

    coordinates = mesh.coordinates()
    cells = mesh.cells()

    for k in range(mesh.num_cells()):
        # Mark current cell
        mf.set_all(0)
        mf[k] = 1

        # Check integration area vs detJ
        # This is not currently implemented in uflacs:
        # x0 = CellOrigin(mesh)
        # But we can extract it from the mesh for a given cell k
        x0 = as_vector(coordinates[cells[k][0]][:])
        # Validate known cell volume A
        assert round(assemble(1.0*dx(1)) - A, 7) == 0.0
        assert round(assemble(1.0/A*dx(1)) - 1.0, 7) == 0.0
        assert round(assemble(A*dx(1)) - A**2, 7) == 0.0
        # Compare abs(detJ) to A
        A2 = Aref*abs(detJ)
        assert round(assemble((A-A2)**2*dx(1)) - 0.0, 7) == 0.0
        assert round(assemble(1.0/A2*dx(1)) - 1.0, 7) == 0.0
        assert round(assemble(A2*dx(1)) - A**2, 7) == 0.0
        # Compare vol to A
        A4 = vol
        assert round(assemble((A-A4)**2*dx(1)) - 0.0, 7) == 0.0
        assert round(assemble((A2-A4)**2*dx(1)) - 0.0, 7) == 0.0
        assert round(assemble(1.0/A4*dx(1)) - 1.0, 7) == 0.0
        assert round(assemble(A4*dx(1)) - A**2, 7) == 0.0

        # Check integral of reference coordinate components over reference
        # tetrahedron:
        Xmp = (1.0/24.0, 1.0/24.0, 1.0/24.0)  # not validated analytically
        for j in range(tdim):
            # Scale by detJ^-1 to get reference cell integral
            assert round(assemble(X[j]/abs(detJ)*dx(1)) - Xmp[j], 7) == 0.0

        # Check average of physical coordinate components over each cell:
        # Compute average of vertex coordinates extracted from mesh
        verts = [coordinates[i][:] for i in cells[k]]
        vavg = sum(verts[1:], verts[0])/len(verts)
        for i in range(gdim):
            # Scale by A^-1 to get average of x, not integral
            assert round(assemble(x[i]/A*dx(1)) - vavg[i], 7) == 0.0

        # Check affine coordinate relations x=x0+J*X, X=K*(x-x0), K*J=I
        x0 = as_vector(coordinates[cells[k][0]][:])
        assert round(assemble((x - (x0+J*X))**2*dx(1)), 7) == 0.0
        assert round(assemble((X - K*(x-x0))**2*dx(1)), 7) == 0.0
        assert round(assemble((K*J - Identity(tdim))**2/A*dx(1)), 7) == 0.0

        # Check cell diameter and circumradius
        assert round(assemble(h/vol*dx(1)) - Cell(mesh, k).h(), 7) == 0.0
        assert round(assemble(R/vol*dx(1)) - Cell(mesh, k).circumradius(), 7) == 0.0


# Some symbolic quantities are only available through uflacs
@skip_in_parallel
def test_triangle_symbolic_geometry(uflacs_representation_only):
    mesh = UnitSquareMesh(1, 1)
    assert mesh.num_cells() == 2
    gdim = mesh.geometry().dim()
    tdim = mesh.topology().dim()

    area = 1.0  # known volume of mesh
    A = area/2.0  # volume of single cell
    Aref = 1.0/2.0  # the volume of the UFC reference triangle

    mf = CellFunction("size_t", mesh)
    dx = Measure("dx", domain=mesh, subdomain_data=mf)

    U0 = FunctionSpace(mesh, "DG", 0)
    V0 = VectorFunctionSpace(mesh, "DG", 0)
    U1 = FunctionSpace(mesh, "DG", 1)
    V1 = VectorFunctionSpace(mesh, "DG", 1)

    # Check coordinates with various consistency checking
    x = SpatialCoordinate(mesh)
    X = CellCoordinate(mesh)
    J = Jacobian(mesh)
    detJ = JacobianDeterminant(mesh)
    K = JacobianInverse(mesh)
    vol = CellVolume(mesh)
    h = CellDiameter(mesh)
    R = Circumradius(mesh)

    coordinates = mesh.coordinates()
    cells = mesh.cells()

    for k in range(mesh.num_cells()):
        # Mark current cell
        mf.set_all(0)
        mf[k] = 1

        # Check integration area vs detJ
        # This is not currently implemented in uflacs:
        # x0 = CellOrigin(mesh)
        # But we can extract it from the mesh for a given cell k
        x0 = as_vector(coordinates[cells[k][0]][:])
        # Validate known cell volume A
        assert round(assemble(1.0*dx(1)) - A, 7) == 0.0
        assert round(assemble(1.0/A*dx(1)) - 1.0, 7) == 0.0
        assert round(assemble(A*dx(1)) - A**2, 7) == 0.0
        # Compare abs(detJ) to A
        A2 = Aref*abs(detJ)
        assert round(assemble((A-A2)**2*dx(1)) - 0.0, 7) == 0.0
        assert round(assemble(1.0/A2*dx(1)) - 1.0, 7) == 0.0
        assert round(assemble(A2*dx(1)) - A**2, 7) == 0.0
        # Compare vol to A
        A4 = vol
        assert round(assemble((A-A4)**2*dx(1)) - 0.0, 7) == 0.0
        assert round(assemble((A2-A4)**2*dx(1)) - 0.0, 7) == 0.0
        assert round(assemble(1.0/A4*dx(1)) - 1.0, 7) == 0.0
        assert round(assemble(A4*dx(1)) - A**2, 7) == 0.0

        # Check integral of reference coordinate components over reference
        # triangle:
        Xmp = (1.0/6.0, 1.0/6.0)
        for j in range(tdim):
            # Scale by detJ^-1 to get reference cell integral
            assert round(assemble(X[j]/abs(detJ)*dx(1)) - Xmp[j], 7) == 0.0

        # Check average of physical coordinate components over each cell:
        # Compute average of vertex coordinates extracted from mesh
        verts = [coordinates[i][:] for i in cells[k]]
        vavg = sum(verts[1:], verts[0])/len(verts)
        for i in range(gdim):
            # Scale by A^-1 to get average of x, not integral
            assert round(assemble(x[i]/A*dx(1)) - vavg[i], 7) == 0.0

        # Check affine coordinate relations x=x0+J*X, X=K*(x-x0), K*J=I
        x0 = as_vector(coordinates[cells[k][0]][:])
        assert round(assemble((x - (x0+J*X))**2*dx(1)), 7) == 0.0
        assert round(assemble((X - K*(x-x0))**2*dx(1)), 7) == 0.0
        assert round(assemble((K*J - Identity(tdim))**2/A*dx(1)), 7) == 0.0

        # Check cell diameter and circumradius
        assert round(assemble(h/vol*dx(1)) - Cell(mesh, k).h(), 7) == 0.0
        assert round(assemble(R/vol*dx(1)) - Cell(mesh, k).circumradius(), 7) == 0.0


xfail_rte = pytest.mark.xfail(raises=RuntimeError, strict=True)

@pytest.mark.parametrize('mesh_factory', [
    (UnitIntervalMesh, (8,)),
    (UnitSquareMesh, (4, 4)),
    (UnitDiscMesh.create, (mpi_comm_world(), 4, 1, 2)),
    (UnitDiscMesh.create, (mpi_comm_world(), 4, 1, 3)),
    (SphericalShellMesh.create, (mpi_comm_world(), 1,)),
    (UnitCubeMesh, (2, 2, 2)),
    (UnitQuadMesh.create, (4, 4)),
    (UnitHexMesh.create, (2, 2, 2)),
    # CellDiameter raises for higher-order meshes
    xfail_rte((UnitDiscMesh.create, (mpi_comm_world(), 4, 2, 2))),
    xfail_rte((UnitDiscMesh.create, (mpi_comm_world(), 4, 2, 3))),
    xfail_rte((SphericalShellMesh.create, (mpi_comm_world(), 2,))),
])
@skip_in_parallel
def test_geometric_quantities(uflacs_representation_only, mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)

    mf = CellFunction('size_t', mesh, 0)
    dx = Measure("dx", domain=mesh, subdomain_data=mf)

    h = CellDiameter(mesh)
    R = Circumradius(mesh)

    for c in cells(mesh):
        # Mark current cell for integration
        mf.set_all(0)
        mf[c] = 1

        # Check cell diameter
        assert numpy.isclose(assemble(h*dx(1))/assemble(1*dx(1)), c.h())

        # Check circumradius if it makes sense
        if mesh.ufl_domain().is_piecewise_linear_simplex_domain():
            assert numpy.isclose(assemble(R*dx(1))/assemble(1*dx(1)), c.circumradius())
