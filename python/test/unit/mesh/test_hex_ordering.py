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
# mesh = dolfin.Mesh(dolfin.MPI.comm_world, dolfin.cpp.mesh.CellType.hexahedron,
#                    points, cells, [], dolfin.cpp.mesh.GhostMode.none)
# dolfin.io.VTKFile("mesh2.pvd").write(mesh)
# print("Here")
S = dolfin.FunctionSpace(mesh, ("Q", 1))

def qf(x):
    return x[:, 0] + 2*x[:, 1] + 3*x[:,2]
cmap = dolfin.fem.create_coordinate_map(mesh.ufl_domain())

mesh.geometry.coord_mapping = cmap
q = dolfin.Function(S)
q.interpolate(qf)
X = S.element.dof_reference_coordinates()
print(X)
print(S.tabulate_dof_coordinates() )
# from IPython import embed; embed()

outfile = dolfin.io.VTKFile("q.pvd")
outfile.write(q)
