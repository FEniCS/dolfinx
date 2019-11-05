import dolfin, dolfin.io
import ufl
import numpy as np
mesh = dolfin.generation.UnitCubeMesh(dolfin.MPI.comm_world, 1, 1, 1,
                                      cell_type=dolfin.cpp.mesh.CellType.hexahedron)
# points =  np.array([[0, 0, 0.], [0, 0, 1.],
#                     [0, 1, 0.], [0, 1, 1.],
#                     [1, 0, 0.], [1, 0, 1.],
#                     [1, 1, 0.], [1, 1, 1.]])
# cells = np.array([range(8)])
points = np.array([[0., 0., 0.], #0
                   [0., 0., 1.],#1
                   [0., 0., 0.5],#2
                   [0., 1., 0.],#3
                   [0., 1., 1.],#4
                   [0., 1., 0.5],#5
                   [0., 0.5, 0.],#6
                   [0., 0.5, 1.],#7
                   [0., 0.5, 0.5],#8
                   [1., 0., 0.],#9
                   [1., 0., 1.],#10
                   [1., 0., 0.5],#11
                   [1., 1., 0.],#12
                   [1., 1., 1.],#13
                   [1., 1., 0.5],#14
                   [1., 0.5, 0.],#15
                   [1., 0.5, 1.],#16
                   [1., 0.5, 0.5],#17#
                   [0.5, 0., 0.],#18
                   [0.5, 0., 1.],#19
                   [0.5, 0., 0.5],#20
                   [0.5, 1., 0.],#21
                   [0.5, 1., 1.],#22
                   [0.5, 1., 0.5],#23
                   [0.5, 0.5, 0.],#24
                   [0.5, 0.5, 1.],#25
                   [0.5, 0.5, 0.5]])#26
cells = np.array([range(27)])
mesh = dolfin.Mesh(dolfin.MPI.comm_world, dolfin.cpp.mesh.CellType.hexahedron,
                   points, cells, [], dolfin.cpp.mesh.GhostMode.none)
# dolfin.io.VTKFile("mesh2.pvd").write(mesh)
# print("Here")
S = dolfin.FunctionSpace(mesh, ("CG", 2))

def qf(x):
     return x[:, 0] + 2*x[:, 1] + 3*x[:,2]*x[:,2]
cmap = dolfin.fem.create_coordinate_map(mesh.ufl_domain())

mesh.geometry.coord_mapping = cmap
q = dolfin.Function(S)
q.interpolate(qf)
# X = S.element.dof_reference_coordinates()
# coord_dofs = mesh.coordinate_dofs().entity_points()
# x_g = mesh.geometry.points
# x = X.copy()

# x_coord_new = np.zeros([1, 2])
# x_coord_new[0] = x_g[coord_dofs[0,2], :2]

outfile = dolfin.io.VTKFile("q.pvd")
outfile.write(q)
import pytest

def test_gmsh_input(order):
    import pygmsh
    # Parameterize test if gmsh gets wider support
    R = 1
    L = 1
    res = 0.2
    algorithm = 2
    element = "hexahedron{0:d}".format(int((order + 1)**3))

    geo = pygmsh.opencascade.Geometry(characteristic_length_min=res, characteristic_length_max=2 * res)
    geo.add_raw_code("Mesh.ElementOrder={0:d};".format(order))
    geo.add_raw_code("Mesh.Format=1;")
    disk = geo.add_disk([0, 0, 0], R)
    rect = geo.add_rectangle([0,-R,0], R, 2 * R)
    union = geo.boolean_difference([disk], [rect])
    geo.add_raw_code("Recombine Surface {1};")
    geo.add_raw_code("Extrude {0, 0," + "{0:.2f}".format(L) + "} {Surface{1}; Curve{6}; Curve{5}; Layers{5}; Recombine;}")

    msh = pygmsh.generate_mesh(geo, verbose=True, dim=3)


    msh_to_dolfin = np.array([0, 4, 10, 3, 7, 15, 9, 17, 22, 1, 5, 12, 2, 6, 14, 11,
                              18, 23, 8, 16, 21, 13, 19, 24, 20, 25, 26])
    cells = np.zeros(msh.cells[element].shape)
    for i in range(len(cells)):
        for j in range(len(msh_to_dolfin)):
            cells[i, j] = msh.cells[element][i, msh_to_dolfin[j]]

    mesh = dolfin.Mesh(dolfin.MPI.comm_world, dolfin.cpp.mesh.CellType.hexahedron, msh.points, cells,
                       [], dolfin.cpp.mesh.GhostMode.none)
    S = dolfin.FunctionSpace(mesh, ("Q", 1))

    def qf(x):
        return x[:, 0] + 2*x[:, 1] + 3*x[:,2]*x[:,2]
    from dolfin.io import VTKFile
    VTKFile("mesh{0:d}.pvd".format(0)).write(mesh)
    surface = dolfin.fem.assemble_scalar(1*ufl.dx(mesh))
    assert dolfin.MPI.sum(mesh.mpi_comm(), surface) == pytest.approx(L / 2 * np.pi * R * R, rel=6e-6)
