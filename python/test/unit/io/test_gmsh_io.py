import numpy as np
import os
import pytest

from mpi4py import MPI as MPI4PY
from dolfin import MPI, cpp
from dolfin.io import VTKFile, HDF5File
from dolfin_utils.test.fixtures import tempdir

assert(tempdir)


@pytest.mark.parametrize("order, element", [(1, "tetra"), (2, "tetra10")])
def test_HDF5_io(tempdir, order, element):
    pytest.importorskip("pygmsh")
    h5py = pytest.importorskip("h5py")

    from pygmsh.opencascade import Geometry
    from pygmsh import generate_mesh

    # Generate a sphere with gmsh with tetrahedral elements
    geo = Geometry()
    geo.add_raw_code("Mesh.Algorithm = 2;")
    geo.add_raw_code("Mesh.Algorithm3D = 10;")
    geo.add_raw_code("Mesh.ElementOrder = {0:d};".format(order))
    geo.add_ball([0, 0, 0], 1, char_length=0.3)
    geo.add_raw_code("Physical Volume (1) = {1};")

    msh = generate_mesh(geo, verbose=False, dim=3)

    # Write gmsh to HDF5
    filename = os.path.join(tempdir, "mesh_order{0:d}.h5".format(order))
    f = h5py.File(filename, "w", driver='mpio', comm=MPI4PY.COMM_WORLD)
    grp = f.create_group("my_mesh")
    grp.create_dataset("cell_indices", data=range(msh.cells[element].shape[0]))
    grp.create_dataset("coordinates", data=msh.points)
    top = grp.create_dataset("topology", data=msh.cells[element])
    top.attrs["celltype"] = np.bytes_('tetrahedron')
    f.close()

    # Read mesh from HDF5
    mesh_file = HDF5File(MPI.comm_world, filename, "r")
    mesh = mesh_file.read_mesh("/my_mesh", False, cpp.mesh.GhostMode.none)
    mesh_file.close()

    # Save mesh with VTK
    outfile = os.path.join(tempdir, "mesh{0:d}.pvd".format(order))
    VTKFile(outfile).write(mesh)
