from enum import Enum
from pathlib import Path

import h5py
import numpy as np

import dolfinx.cpp as _cpp
from dolfinx.mesh import Mesh


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
    pyramid = 14
    wedge = 73

    def __int__(self) -> int:
        return self.value


vtk_to_str = {
    VTKCellType.line: "interval",
    VTKCellType.triangle: "triangle",
    VTKCellType.quadrilateral: "quadrilateral",
    VTKCellType.tetrahedron: "tetrahedron",
    VTKCellType.hexahedron: "hexahedron",
    VTKCellType.vertex: "point",
    VTKCellType.wedge: "prism",
    VTKCellType.pyramid: "pyramid",
}


def str_to_vtk(name):
    return next(key for (key, value) in vtk_to_str.items() if value == name)


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

        # Extract topology information for each cell type
        cell_types = mesh.topology.entity_types[mesh.topology.dim]
        cell_index_maps = [imap for imap in mesh.topology.index_maps(mesh.topology.dim)]
        num_cells = [cmap.size_local for cmap in cell_index_maps]
        num_cells_global = [cmap.size_global for cmap in cell_index_maps]

        # Geometry dofmap and points
        geom_imap = mesh.geometry.index_map()
        gdim = mesh.geometry.dim
        geom_global_shape = (geom_imap.size_global, gdim)
        gdtype = mesh.geometry.x.dtype
        geom_irange = geom_imap.local_range

        # Create dataset for storing the nodes of the geometry
        p_string = np.bytes_("Points")
        geom_set = hdf.create_dataset(p_string, geom_global_shape, dtype=gdtype)
        geom_set[geom_irange[0] : geom_irange[1], :] = mesh.geometry.x[: geom_imap.size_local, :]

        # Metadata for number of nodes
        num_points = hdf.create_dataset("NumberOfPoints", (1,), dtype=np.int64)
        num_points[0] = geom_imap.size_global

        # VTKHDF5 stores the cells as an adjacency list, where cell types might be jumbled up.
        topology_flattened = []
        topology_num_cell_points = []
        for i in range(len(cell_index_maps)):
            g_dofmap = mesh.geometry.dofmaps(i)
            local_dm = g_dofmap[: cell_index_maps[i].size_local, :].copy()
            # Permute DOLFINx order to VTK
            map_vtk = np.argsort(_cpp.io.perm_vtk(cell_types[i], local_dm.shape[1]))
            local_dm = local_dm[:, map_vtk]
            global_dm = geom_imap.local_to_global(local_dm.flatten())
            topology_flattened.append(global_dm)
            topology_num_cell_points.append(
                np.full(g_dofmap.shape[0], g_dofmap.shape[1], dtype=np.int32)
            )
        topo_offsets = np.cumsum(np.hstack(topology_num_cell_points))

        num_nodes_per_cell = [mesh.geometry.cmaps(i).dim for i in range(len(num_cells))]
        cell_start_position = [cmap.local_range[0] for cmap in cell_index_maps]
        cell_stop_position = [cmap.local_range[1] for cmap in cell_index_maps]

        # Compute overall cell offset from offsets for each cell type
        offset_start_position = sum(cell_start_position)
        offset_stop_position = sum(cell_stop_position)

        # Compute overall topology offset from offsets for each cell type
        topology_start = np.dot(num_nodes_per_cell, cell_start_position)
        topo_offsets += topology_start

        # Offsets into topology
        offsets = hdf.create_dataset("Offsets", (sum(num_cells_global) + 1,), dtype=np.int64)
        if comm.rank == 0:
            offsets[0] = 0
        offsets[offset_start_position + 1 : offset_stop_position + 1] = topo_offsets

        # Store global mesh connectivity
        topology_size_global = np.dot(num_cells_global, num_nodes_per_cell)
        topology_set = hdf.create_dataset("Connectivity", (topology_size_global,), dtype=np.int64)
        topo_out = np.hstack(topology_flattened)
        topology_set[topology_start : topology_start + len(topo_out)] = topo_out

        # Store cell types
        type_set = hdf.create_dataset("Types", (sum(num_cells_global),), dtype=np.uint8)
        cts = np.hstack(
            [
                np.full(
                    cell_index_maps[i].size_local,
                    int(str_to_vtk(cell_types[i].name)),
                    dtype=np.uint8,
                )
                for i in range(len(cell_index_maps))
            ]
        )
        type_set[offset_start_position:offset_stop_position] = cts

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
