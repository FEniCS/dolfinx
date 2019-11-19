#
# .. _demo_mesh_io:
#
# Mesh Input/Output
# ================
#
# This demo will illustrate how to:
#
# * Generate a mesh using pygmsh
# * Generate meshes consisting of triangle, tetrahedron, quadrilateral or
#   hexahedral elements
# * Generate meshes of different orders (1, 2 and 3)
# * Ways of reading meshes into dolfinx
# * Ways save output
#
# First, we illustrate how to generate 2D problems with pygmsh, for both
# triangular and quadrilateral elements.
# We start by importing the opencascade backend of gmsh, as well as the mesh
# generator.

from pygmsh.opencascade import Geometry
from pygmsh import generate_mesh

# Then, we define a function, which takes in the element type, and which order
# of the element we would like to generate. The options are:
#
# * First to third order quadrilateral elements
# * First to third order triangular elements
def generate_pygmsh_mesh(element, order, filename, R=1, res=0.3, outdir="output/"):
    """
    Generates a circular mesh with radius one and mesh resolution 0.3.
    The function returns a  :py:class:`Mesh <mesio._mesh.Mesh>` object,
    which contains the points defining the mesh, their connectivity and
    physical markers of the cells and facets.
    """
    geo = Geometry()
    # Select mesh order
    geo.add_raw_code("Mesh.ElementOrder = {0:d};".format(order))

    disk = geo.add_disk([0, 0, 0], R, char_length=res)
    if "quad" in element:
        # Choose meshing algorithm for quadrilaterals (Frontal-Delaunay for Quads)
        geo.add_raw_code("Mesh.Algorithm = 8;")
        # Recombine surface to obtain quadrilateral elements
        geo.add_raw_code("Recombine Surface {1};")
        # Guarantee full-quad mesh
        geo.add_raw_code("Mesh.RecombinationAlgorithm = 2;")

    # Add physical markers for the whole boundary, with value two,
    # and a physical cell marker with value one for all interior cells
    geo.add_raw_code("Physical Surface (1) = {1};")
    geo.add_raw_code("Physical Curve (2) = {1};")

    # When the mesh is generated, the z-coordinates are removed so that it
    # is no longer a 2D mesh embedded in a 3D space.
    msh = generate_mesh(geo, verbose=False, dim=2, prune_z_0=True,
                        geo_filename=outdir + filename+ ".geo")
    return msh

# The next step is to save the Mesh to file.

def save_mesh_to_file(mesh, element, order, filename, outdir="output/"):
    """
    Saving the meshio Mesh to HDF5, readable by the dolfinx paraview plugin.
    """
    # For third order meshes, the best way of saving them is to use HDF5
    facets = {1:"line", 2:"line3", 3: "line4"}

    import h5py
    import mpi4py
    import numpy
    if order == 3 and "quad" in element:
        # Gmsh uses a custom msh ordering of quadrilaterals of order three. We permute these to VTK.
        msh_to_vtk = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 11, 10, 12, 13, 15, 14])
        cells = numpy.zeros(mesh.cells[element].shape)
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                cells[i,j] = mesh.cells[element][i, msh_to_vtk[j]]
    else:
        cells = mesh.cells[element]
    # The dolfin HDF5 storage format requires the following groups, datasets and attributes
    outfile = h5py.File(outdir + filename + ".h5", "w", driver='mpio', comm=mpi4py.MPI.COMM_WORLD)
    grp = outfile.create_group("mesh")
    grp.create_dataset("cell_indices", data=range(cells.shape[0]))
    grp.create_dataset("coordinates", data=mesh.points)
    top = grp.create_dataset("topology", data=cells)
    cell_type_HDF5 = "triangle" if "triangle" in element else "quadrilateral"
    top.attrs["celltype"] = numpy.bytes_(cell_type_HDF5)

    cell_func = outfile.create_group("cellfunction")
    c_top = cell_func.create_dataset("topology", data=cells)
    cell_func.create_dataset("coordinates", data=mesh.points)
    cell_func.create_dataset("values", data=mesh.cell_data[element]["gmsh:physical"])
    c_top.attrs["celltype"] = numpy.bytes_(cell_type_HDF5)

    facet_func = outfile.create_group("facetfunction")
    facet_func.attrs["dimension"] = 1
    f_top = facet_func.create_dataset("topology", data=mesh.cells[facets[order]])
    facet_func.create_dataset("coordinates", data=mesh.points)
    facet_func.create_dataset("values", data=mesh.cell_data[facets[order]]["gmsh:physical"])

    outfile.close()

def load_to_dolfin(element, order, filename, outdir="output/"):
    """
    Function illustrating how to load the saved meshes to dolfin, and write the
    to a format that can be visualized in Paraview
    """
    from dolfin import MPI, cpp
    from dolfin.cpp.mesh import GhostMode
    from dolfin.io import VTKFile
    # if order < 3 and element == "triangle":
    #     from dolfin.io import XDMFFile
    #     with XDMFFile(MPI.comm_world, filename + ".xdmf") as xdmf:
    #         mesh = xdmf.read_mesh(GhostMode.none)
    # else:
    # Load the HDF5 file
    from dolfin.io import HDF5File
    mesh_file = HDF5File(MPI.comm_world, outdir + filename + ".h5".format(element, order), "r")
    # We did not save partition instructions in the HDF5-file
    mesh = mesh_file.read_mesh("/mesh", False, GhostMode.none)
    # Load the cell function:

    cf = mesh_file.read_mf_double(mesh, "/cellfunction")
    # need to figure out how one can read data from only fractions of the elements.
    mvc = mesh_file.read_mvc_size_t(mesh, "/facetfunction")
    mesh_file.close()

    VTKFile(outdir + "cf" + filename[4:] + ".pvd".format(element, order)).write(cf)

    # Save mesh with VTK
    VTKFile((outdir + filename + ".pvd").format(element, order)).write(mesh)

    ff = cpp.mesh.MeshFunctionSizet(mesh, mvc, 1)
    print(ff.values)
    VTKFile(outdir + "ff" + filename[4:] + ".pvd".format(element, order)).write(ff)

element_names = {"triangle": {1: "triangle", 2: "triangle6", 3: "triangle10"},
                 "quadrilateral": {1: "quad", 2: "quad9", 3: "quad16"}}
for order in [1, 2, 3]:
    for element in element_names.keys():
        filename = "mesh_{0:s}_{1:d}".format(element, order)
        ele = element_names[element][order]
        meshio_mesh = generate_pygmsh_mesh(ele, order, filename)

        save_mesh_to_file(meshio_mesh, ele , order, filename)
        load_to_dolfin(ele, order, filename)
