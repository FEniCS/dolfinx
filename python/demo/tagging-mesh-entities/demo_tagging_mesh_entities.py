#
# .. _demo_tagging_mesh_entities:
#
# Tagging mesh entities
# =====================
#
# This demo is implemented in a single Python file,
# :download:`demo_tagging_mesh_entities.py`, which
# describes how to read the boundaries from a gmsh
# mesh('.msh') into dolfin.
#
# This demo illustrates how to:
#
# * Tag different regions in a mesh.
# * Convert the mesh to XDMF using meshio.
# * Read the mesh along with tag information into dolfin.
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
# :py:mod:`meshio` and :py:mod:`dolfin` is imported: ::

from pygmsh import generate_mesh
from pygmsh.built_in.geometry import Geometry
import meshio
from dolfin import MPI, cpp
from dolfin.io import XDMFFile

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

# Now we have the data related to the mesh available with us. The next step is
# to convert the data into `XDMF` format that is supported in dolfin. For this
# we make use of the package :py:mod:`meshio`. Note that if you have created
# the mesh using `gmsh`, then you first need to read the mesh into meshio and
# then convert it to `XDMF`. ::

# -----------------Step - 2 - Convert mesh --------------
meshio.write("poisson_subdomain.xdmf", meshio.Mesh(
    points=points,
    cells={"triangle": cells["triangle"]},
    field_data=field_data))

# The next step is to read the mesh in dolfin. ::

# -----------------Step - 3 - Read mesh -----------------
with XDMFFile(MPI.comm_world,
              "poisson_subdomain.xdmf") as xdmf_infile:
    mesh = xdmf_infile.read_mesh(cpp.mesh.GhostMode.none)
    tag_information = xdmf_infile.read_information_int()
    print(tag_information)

# This way we can read the mesh file created via `gmsh` along with the information
# regarding tagged regions into `dolfin`. The tag_information read by this demo
# program is the one used for the Poisson Subdomain demo program.
