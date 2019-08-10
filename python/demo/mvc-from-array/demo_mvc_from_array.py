#
# .. _demo_mvc_from_array:
#
# Mesh Value Collection from arrays
# =================================
#
# This demo is implemented in a single Python file,
# :download:`demo_mvc_from_array.py`, which
# describes how to construct a mesh value collection directly from
# pygmsh variables.
#
# This demo illustrates how to:
#
# * Tag different regions in a mesh using pygmsh.
# * Construct mesh value collection from pygmsh variables.
#
# The domain under consideration in this demo will look as follows:
#
# .. image:: poisson_subdomain.png
#    :scale: 40 %
#
#
# Implementation
# --------------
#
# First, we need to create the domain under consideration in gmsh
# and then tag the boundaries with name tags as presented in the
# image above. For information on how to create a mesh in gmsh and
# how to tag the boundaries you should read the official documentation of gmsh.
#
# For the sake of completeness we start by defining the mesh by using the
# python package pygmsh. First, the :py:mod:`pygmsh` module along with
# :py:mod:`dolfin` is imported: ::

from pygmsh import generate_mesh
from pygmsh.built_in.geometry import Geometry
from dolfin import MPI, cpp, MeshValueCollection

# We begin by constructing the mesh using
# :py:mod:`pygmsh` and then marking the different mesh regions using the
# :py:class:`add_physical()` method. This simulates the construction and
# tagging of mesh in gmsh ::

# -----------------Step - 1 - Define mesh --------------
geom = Geometry()

mesh_ele_size = 0.2
p1 = geom.add_point([0.0, 0.0, 0], lcar=mesh_ele_size)
p2 = geom.add_point([0.0, 1.0, 0], lcar=mesh_ele_size)
p3 = geom.add_point([1.0, 1.0, 0], lcar=mesh_ele_size)
p4 = geom.add_point([1.0, 0.0, 0], lcar=mesh_ele_size)
p5 = geom.add_point([0.2, 0.5, 0], lcar=mesh_ele_size)
p6 = geom.add_point([0.2, 0.7, 0], lcar=mesh_ele_size)
p7 = geom.add_point([1.0, 0.5, 0], lcar=mesh_ele_size)
p8 = geom.add_point([1.0, 0.7, 0], lcar=mesh_ele_size)

l1 = geom.add_line(p1, p4)
l2 = geom.add_line(p3, p2)
l3 = geom.add_line(p2, p1)
l4 = geom.add_line(p7, p5)
l5 = geom.add_line(p5, p6)
l6 = geom.add_line(p6, p8)
l7 = geom.add_line(p4, p7)
l8 = geom.add_line(p7, p8)
l9 = geom.add_line(p8, p3)

ll1 = geom.add_line_loop(lines=[l2, l3, l1, l7, l4, l5, l6, l9])
ps1 = geom.add_plane_surface(ll1)

ll2 = geom.add_line_loop(lines=[l6, -l8, l4, l5])
ps2 = geom.add_plane_surface(ll2)

# Tag line and surface
geom.add_physical(l3, label="LEFT")
geom.add_physical(l2, label="TOP")
geom.add_physical([l9, l8, l7], label="RIGHT")
geom.add_physical(l1, label="BOTTOM")

geom.add_physical(ps1, label="DOMAIN")
geom.add_physical(ps2, label="OBSTACLE")

msh = generate_mesh(geom)
points, cells = msh.points, msh.cells
cell_data, field_data = msh.cell_data, msh.field_data

# Now we have the data related to the mesh available with us.
# We can now directly create mesh and mesh value collection using this data. ::


# -----------------Step - 2 - Construct mesh in dolfin -----------------------
mesh = cpp.mesh.Mesh(
    MPI.comm_world,
    cpp.mesh.CellType.triangle,
    points[:, :2],  # Converting to 2D
    cells["triangle"],
    [],
    cpp.mesh.GhostMode.none,
)

# ----------------Step - 3 - Construct mesh value collection -----------------
mvc_boundaries = MeshValueCollection(
    "size_t",
    mesh,
    1,
    cells["line"].tolist(),
    cell_data["line"]["gmsh:physical"].tolist(),
)

mvc_subdomain = MeshValueCollection(
    "size_t",
    mesh,
    2,
    cells["triangle"].tolist(),
    cell_data["triangle"]["gmsh:physical"].tolist(),
)

tag_info = {key: field_data[key][0] for key in field_data}

# The mesh value collection, mvc_subdomain and
# mvc_boundaries contains information about the integer tag of tagged
# elements. The variable tag_info is a dictionary where the key is string tag
# and the value is the corresponding int tag. We would use this dictionary to
# specify Measures to integrate over specified regions and to specify
# different boundary conditions. The above mesh value collections are defined
# with the sole purpose of populating mesh functions.


# ----------------Step - 4 - Construct mesh function -------------------------
mf_triangle = cpp.mesh.MeshFunctionSizet(mesh, mvc_subdomain, 0)
mf_line = cpp.mesh.MeshFunctionSizet(mesh, mvc_boundaries, 0)

print(mf_line.values)

# This way we can directly construct mesh and mesh value collection from
# pygmsh variables. The tag_info read by this demo program is the one used
# for the Poisson Subdomain demo program.
