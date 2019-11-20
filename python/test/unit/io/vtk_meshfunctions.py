from dolfin import MPI, UnitSquareMesh, MeshFunction, cpp, UnitCubeMesh
from dolfin.io import VTKFile
# mesh = cpp.generation.UnitDiscMesh.create(MPI.comm_world, 3,
#                                           cpp.mesh.GhostMode.none)
# mesh = UnitSquareMesh(MPI.comm_world, 2, 2)#, cpp.mesh.CellType.quadrilateral)

mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2, cpp.mesh.CellType.hexahedron)
for d in range(mesh.topology.dim + 1):
    mf = MeshFunction("size_t", mesh, mesh.topology.dim - d, 1)
    f = VTKFile("mf{0:d}.pvd".format(mesh.topology.dim - d))
    mesh.create_connectivity_all()
    f.write(mf, 0.)
