from enum import Enum
from pathlib import Path

import h5py
import numpy as np

import dolfinx.cpp as _cpp
from dolfinx.mesh import Mesh

# # Copy from dolfinx::io::xdmf_utils.cpp
# xdmf_to_dolfin = {
#     "polyvertex": ("point", 1),
#     "polyline": ("interval", 1),
#     "edge_3": ("interval", 2),
#     "triangle": ("triangle", 1),
#     "triangle_6": ("triangle", 2),
#     "tetrahedron": ("tetrahedron", 1),
#     "tetrahedron_10": ("tetrahedron", 2),
#     "quadrilateral": ("quadrilateral", 1),
#     "quadrilateral_9": ("quadrilateral", 2),
#     "quadrilateral_16": ("quadrilateral", 3),
#     "hexahedron": ("hexahedron", 1),
#     "wedge": ("prism", 1),
#     "hexahedron_27": ("hexahedron", 2),
# }


class VTKCellType(Enum):
    """
    VTK Cell types (for arbitrary order Lagrange):
    https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    """

    vertex = 2
    line = 68
    triangle = 69
    quadrilateral = 70
    tetrahedron = 78
    hexahedron = 72
    wedge = 73

    def __str__(self) -> str:
        if self == VTKCellType.line:
            return "interval"
        elif self == VTKCellType.triangle:
            return "triangle"
        elif self == VTKCellType.quadrilateral:
            return "quadrilateral"
        elif self == VTKCellType.tetrahedron:
            return "tetrahedron"
        elif self == VTKCellType.hexahedron:
            return "hexahedron"
        elif self == VTKCellType.vertex:
            return "point"
        elif self == VTKCellType.wedge:
            return "prism"
        else:
            raise ValueError(f"Unknown cell type: {self}")

    @classmethod
    def to_vtk(self, cell):
        if cell == "interval":
            return VTKCellType.line
        elif cell == "triangle":
            return VTKCellType.triangle
        elif cell == "quadrilateral":
            return VTKCellType.quadrilateral
        elif cell == "tetrahedron":
            return VTKCellType.tetrahedron
        elif cell == "hexahedron":
            return VTKCellType.hexahedron
        elif cell == "point":
            return VTKCellType.vertex
        elif cell == "prism":
            return VTKCellType.wedge
        else:
            raise ValueError(f"Unknown cell type: {cell}")

    def __int__(self) -> int:
        return self.value


def write(mesh: Mesh, filename: str | Path):
    """
    Write mesh to VTK HDF5 format
    """
    comm = mesh.comm
    fname = Path(filename).with_suffix(".vtkhdf")
    with h5py.File(fname, "w", driver="mpio", comm=comm) as inf:
        # Store data that would be required when reading in the mesh again
        metadata = inf.create_group(np.bytes_("Metadata"))
        metadata.attrs["gdim"] = mesh.geometry.dim

        # Create VTKHDF group
        hdf = inf.create_group(np.bytes_("VTKHDF"))
        h5py.string_dtype(encoding="ascii")
        hdf.attrs["Version"] = [2, 2]
        hdf.attrs["Type"] = np.bytes_("UnstructuredGrid")

        # Extract topology information
        cell_index_maps = [imap for imap in mesh.topology.index_maps(mesh.topology.dim)]
        num_cells = [cmap.size_local for cmap in cell_index_maps]
        num_cells_global = np.array([cmap.size_global for cmap in cell_index_maps], dtype=np.int64)

        # Extract various ownership information from the geometry
        geom_imap = mesh.geometry.index_map()
        gdim = mesh.geometry.dim
        geom_global_shape = (geom_imap.size_global, gdim)
        gdtype = mesh.geometry.x.dtype
        geom_irange = geom_imap.local_range
        c_els = [mesh.geometry.cmaps(i) for i in range(len(num_cells))]

        # Create dataset for storing the nodes of the geometry
        p_string = np.bytes_("Points")
        geom_set = hdf.create_dataset(p_string, geom_global_shape, dtype=gdtype)
        geom_set[geom_irange[0] : geom_irange[1], :] = mesh.geometry.x[: geom_imap.size_local, :]

        entity_cells = mesh.topology.entity_types[mesh.topology.dim]

        # VTKHDF5 stores the geometry as an adjacency lists, where cell types might be jumbled up.
        geometry_flattened = []
        geometry_num_cell_dofs = []
        for i in range(len(cell_index_maps)):
            g_dofmap = mesh.geometry.dofmaps(i)
            local_dm = g_dofmap[: cell_index_maps[i].size_local, :].copy()
            # Permute DOLFINx order to VTK
            map_vtk = np.argsort(_cpp.io.perm_vtk(entity_cells[i], local_dm.shape[1]))
            local_dm = local_dm[:, map_vtk]
            global_dm = geom_imap.local_to_global(local_dm.flatten())
            geometry_flattened.append(global_dm)
            geometry_num_cell_dofs.append(
                np.full(g_dofmap.shape[0], g_dofmap.shape[1], dtype=np.int32)
            )
        geom_offsets = np.zeros(sum(num_cells) + 1)
        geom_offsets[1:] = np.cumsum(np.hstack(geometry_num_cell_dofs))

        num_nodes_per_cell = [c_el.dim for c_el in c_els]
        cell_start_position = [cmap.local_range[0] for cmap in cell_index_maps]
        cell_stop_position = [cmap.local_range[1] for cmap in cell_index_maps]

        offset_start_position = sum(cell_start_position)
        offset_stop_position = sum(cell_stop_position)
        # Adapt offset for multiple processes
        accumulated_other_proc_topology = [
            ni * csp for (ni, csp) in zip(num_nodes_per_cell, cell_start_position)
        ]
        geom_offsets += sum(accumulated_other_proc_topology)

        # If rank 0 we start at 0, otherwise we start one further than the number
        # of cells on the previous rank
        process_start_shift = 0 if mesh.comm.rank == 0 else 1
        offset_start_position += process_start_shift
        process_stop_shift = 1
        offset_stop_position += process_stop_shift

        # Offsets
        offsets = hdf.create_dataset("Offsets", (sum(num_cells_global) + 1,), dtype=np.int64)
        offsets[offset_start_position:offset_stop_position] = geom_offsets[process_start_shift:]

        # Store global mesh connectivity for geometry
        topology_size_global = sum(
            [ncg * nnpc for (ncg, nnpc) in zip(num_cells_global, num_nodes_per_cell)]
        )
        top_set = hdf.create_dataset("Connectivity", (topology_size_global,), dtype=np.int64)
        geom_out = np.hstack(geometry_flattened)
        top_start_pos = sum(accumulated_other_proc_topology)
        top_set[top_start_pos : top_start_pos + len(geom_out)] = geom_out

        # Store cell types
        type_set = hdf.create_dataset("Types", (sum(num_cells_global),), dtype=np.uint8)
        cts = np.hstack(
            [
                np.full(
                    cell_index_maps[i].size_local,
                    int(VTKCellType.to_vtk(entity_cells[i].name)),
                    dtype=np.uint8,
                )
                for i in range(len(cell_index_maps))
            ]
        )
        type_set[sum(cell_start_position) : sum(cell_stop_position)] = cts
        # Geom dofmap offset
        con_part = hdf.create_dataset("NumberOfConnectivityIds", (1,), dtype=np.int64)
        con_part[0] = topology_size_global

        # Store meta-variable used by VTKHDF5 when partitioning data. We do not partition data,
        # as then everything becomes "local index", and cannot be read in again
        # in parallel on M processes.
        hdf.create_dataset(
            "NumberOfCells",
            (1,),
            dtype=np.int64,
            data=np.array([sum(num_cells_global)], dtype=np.int64),
        )

        # Similar metadata for number of nodes
        num_points = hdf.create_dataset("NumberOfPoints", (1,), dtype=np.int64)
        num_points[0] = geom_imap.size_global
