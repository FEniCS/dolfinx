import dolfinx
from dolfinx.mesh import MeshTags
from mpi4py import MPI
import numpy as np
from IPython import embed

mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 2, 1)
tdim = mesh.topology.dim
num_cells = mesh.topology.index_map(tdim).size_local
indices = np.arange(num_cells, dtype=np.int32)
values = indices
ct = dolfinx.MeshTags(mesh, tdim, indices[2:], values[2:])
#V = dolfinx.FunctionSpace(ct, ("CG", 1))


class MeshView():

    def __init__(self, meshtag: dolfinx.MeshTags):
        mesh = meshtag.mesh
        mesh.topology.create_connemeshtagivity(meshtag.dim, 0)
        c_to_v = mesh.topology.connectivity(meshtag.dim, 0)
        vertices = []
        for entity in meshtag.indices:
            vertices_e = c_to_v.links(entity)
            [vertices.append(vertex) for vertex in vertices_e]
        unique_vertices = np.unique(vertices)

        # Create index map for vertices and cells of MeshView
        # FIXME: The last three arrays needs updating in parallel
        imap = dolfinx.cpp.common.IndexMap(MPI.COMM_WORLD, unique_vertices.size, [], [], [])
        # FIXME: strip ghosts
        imap_c = dolfinx.cpp.common.IndexMap(MPI.COMM_WORLD, len(meshtag.indices), [], [], [])

        # Create map from mesh to MeshView for vertices
        l_vertices = np.arange(imap.local_range[0], imap.local_range[1])
        vertex_map_to_view = {}
        for index in l_vertices:
            vertex_map_to_view[unique_vertices[index]] = index

        # Map vertices of each cell in MeshView to new IndexMap
        new_cell_dofs = np.array([], dtype=np.int32)
        new_cell_dofs_offset = [0]
        for entity in meshtag.indices:
            vertices = c_to_v.links(entity)
            view_vertices = np.array([vertex_map_to_view[vertex] for vertex in vertices], dtype=np.int32)
            new_cell_dofs = np.hstack([new_cell_dofs, view_vertices])
            new_cell_dofs_offset.append(len(new_cell_dofs))
        new_connectivity = dolfinx.cpp.graph.AdjacencyList_int32(new_cell_dofs, new_cell_dofs_offset)

        print(c_to_v)
        print(new_connectivity)

        # Create topology of MeshView
        new_top = dolfinx.cpp.mesh.Topology(MPI.COMM_WORLD, mesh.topology.cell_type)

        # Create vertex to vertex map (is identity)
        v_to_v = dolfinx.cpp.graph.AdjacencyList_int32(
            np.arange(len(unique_vertices), dtype=np.int32), np.arange(len(unique_vertices), dtype=np.int32))
        new_top.set_connectivity(v_to_v, 0, 0)
        # Set Index map for vertices
        new_top.set_index_map(0, imap)

        # Set mesh connectivity for cells and index map for cells
        new_top.set_connectivity(new_connectivity, meshtag.dim, 0)
        new_top.set_index_map(meshtag.dim, imap_c)
        self.topology = new_top
        self.mesh = mesh
        self.cell_map = meshtag.indices
        self.dim = meshtag.dim


mv = MeshView(ct)
num_cells_local = mv.topology.index_map(mv.dim).size_local
for cell in range(num_cells_local):
    print(cell, mv.cell_map[cell], mv.mesh.topology.connectivity(mv.cell_map[cell]))
# embed()
mesh.topology.create_connectivity(ct.dim, 0)
c_to_v = mesh.topology.connectivity(ct.dim, 0)
print(c_to_v)

# new_top.create_connectivity(tdim - 1, 0)
# new_f_to_v = new_top.connectivity(tdim - 1, 0)
# print(new_f_to_v)
