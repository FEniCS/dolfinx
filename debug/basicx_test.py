import time
import ufl
from IPython import embed
import dolfinx.io
import basix
import dolfinx
import dolfinx.geometry
import gmsh
import numpy as np
from dolfinx.cpp.io import perm_gmsh
from dolfinx.cpp.mesh import to_type
from dolfinx.io import (extract_gmsh_geometry,
                        extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh)
from dolfinx.mesh import create_mesh
from mpi4py import MPI

gmsh.initialize()
rect_tag = gmsh.model.occ.addRectangle(0, -1, 0, 5, 1)
gdim = 2
r = 0.4
c_x, c_y = 2.5, 0.5
#rect_tag = gmsh.model.occ.addDisk(c_x, -c_y, 0, r, r)

circ_tag = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [rect_tag], rect_tag)
gmsh.model.addPhysicalGroup(2, [circ_tag], circ_tag)
# View gmsh output
# gmsh.option.setNumber("General.Terminal", 1)

gmsh.model.mesh.field.add("Distance", 1)
circle_arc = gmsh.model.getBoundary([(2, circ_tag)])
gmsh.model.mesh.field.setNumbers(1, "EdgesList", [e[1] for e in circle_arc])
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "IField", 1)
gmsh.model.mesh.field.setNumber(2, "LcMin", r / 400)  # r / 4)
gmsh.model.mesh.field.setNumber(2, "LcMax", r / 100)  # r)
gmsh.model.mesh.field.setNumber(2, "DistMin", r / 4)
gmsh.model.mesh.field.setNumber(2, "DistMax", r / 1.5)
gmsh.model.mesh.field.setAsBackgroundMesh(2)
gmsh.model.mesh.generate(dim=gdim)

# Convert mesh to dolfin-X
cell_information = {}
topologies, cell_dimensions = None, None
if MPI.COMM_WORLD.rank == 0:
    # Get mesh geometry
    x = extract_gmsh_geometry(gmsh.model)

    # Get mesh topology for each element
    topologies = extract_gmsh_topology_and_markers(gmsh.model)

    # Get information about each cell type from the msh files
    num_cell_types = len(topologies.keys())
    cell_dimensions = np.zeros(num_cell_types, dtype=np.int32)
    for i, element in enumerate(topologies.keys()):
        properties = gmsh.model.mesh.getElementProperties(element)
        name, dim, order, num_nodes, local_coords, _ = properties
        cell_information[i] = {"id": element, "dim": dim,
                               "num_nodes": num_nodes}
        cell_dimensions[i] = dim

gmsh.finalize()
if MPI.COMM_WORLD.rank == 0:
    # Sort elements by ascending dimension
    perm_sort = np.argsort(cell_dimensions)

    # Broadcast cell type data and geometric dimension
    cell_id = cell_information[perm_sort[-1]]["id"]
    tdim = cell_information[perm_sort[-1]]["dim"]
    num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
    cell_id, num_nodes = MPI.COMM_WORLD.bcast([cell_id, num_nodes], root=0)
    cells = topologies[cell_id]["topology"]
    cell_values = topologies[cell_id]["cell_data"]

else:
    cell_id, num_nodes = MPI.COMM_WORLD.bcast([None, None], root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, gdim])
    cell_values = np.empty((0,))


# Create distributed mesh
ufl_domain = ufl_mesh_from_gmsh(cell_id, gdim)
gmsh_cell_perm = perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
cells = cells[:, gmsh_cell_perm]
mesh = create_mesh(MPI.COMM_WORLD, cells, x[:, :gdim], ufl_domain)


def curved_contact(x):
    # Curved contact area
    return np.logical_and(np.isclose((x[0] - c_x)**2 + (x[1] - c_y)**2, r**2), x[1] < c_y - 0.1 * r)


tdim = mesh.topology.dim
fdim = tdim - 1
circ_vertices = dolfinx.mesh.locate_entities_boundary(mesh, 0, curved_contact)
values = np.full(len(circ_vertices), 2, dtype=np.int32)
ft = dolfinx.MeshTags(mesh, 0, circ_vertices, values)
ft.name = "Vertex tags"


def master_obstacle(x):
    return x[1] <= 0


rect_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, master_obstacle)
mesh.topology.create_connectivity(fdim, tdim)
f_to_c = mesh.topology.connectivity(fdim, tdim)
master_cells = np.zeros(len(rect_facets), dtype=np.int32)
for i, facet in enumerate(rect_facets):
    r_cells = f_to_c.links(facet)
    assert(len(r_cells) == 1)
    master_cells[i] = r_cells[0]

ct = dolfinx.MeshTags(mesh, tdim, master_cells, np.full(len(master_cells), 1, dtype=np.int32))
ct.name = "Cell tags"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct)
    xdmf.write_meshtags(ft)


# quadrature_rule = basix.make_quadrature("default", basix.CellType.triangle, 2)

imap = mesh.topology.index_map(mesh.topology.dim)
start = time.time()
#master_cells = range(imap.size_local + imap.num_ghosts)
master_bbox = dolfinx.cpp.geometry.BoundingBoxTree(mesh, tdim, master_cells)
midpoint_tree = dolfinx.geometry.create_midpoint_tree(mesh, mesh.topology.dim, master_cells)
end = time.time()
print(len(master_cells))
tot = end - start
print("Num cells:", mesh.topology.index_map(mesh.topology.dim).size_local)
print("Midpointtree init (avg) {0:.2e}".format(MPI.COMM_WORLD.allreduce(tot, op=MPI.SUM) / MPI.COMM_WORLD.size))

geometry_nodes = dolfinx.cpp.mesh.entities_to_geometry(mesh, 0, circ_vertices, False).T[0]
points = mesh.geometry.x[geometry_nodes[0]]
#start = time.time()
closest_cell, dist = dolfinx.cpp.geometry.compute_closest_entity(master_bbox, midpoint_tree, mesh, points)
#end = time.time()
#print("Runtime = {0:.2e}".format(end - start))
# unique_cells = np.unique(closest_cell)
# ct2 = dolfinx.MeshTags(mesh, tdim, unique_cells, np.full(len(unique_cells), 1, dtype=np.int32))
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_meshtags(ct2)
#     xdmf.write_meshtags(ft)


# for (x, cell, d) in zip(points, closest_cell, dist):
#     cell_nodes = dolfinx.cpp.mesh.entities_to_geometry(mesh, tdim, [cell], False)[0]
#     cell_geo = mesh.geometry.x[cell_nodes]
#     actual_distance = dolfinx.cpp.geometry.compute_distance_gjk(x, cell_geo)
#     #print(actual_distance, np.linalg.norm(d), d)
