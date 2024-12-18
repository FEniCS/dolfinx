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
        num_cells = sum([imap.size_global for imap in cell_index_maps])

        # Extract various ownership information from the geometry
        geom_imap = mesh.geometry.index_map()
        gdim = mesh.geometry.dim
        geom_global_shape = (geom_imap.size_global, gdim)
        gdtype = mesh.geometry.x.dtype
        geom_irange = geom_imap.local_range

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
            # FIXME: add local to global map here
            geometry_flattened.append(local_dm.flatten())
            geometry_num_cell_dofs.append(
                np.full(g_dofmap.shape[0], g_dofmap.shape[1], dtype=np.int32)
            )
        geom_offsets = np.zeros(num_cells + 1)
        geom_offsets[1:] = np.cumsum(np.hstack(geometry_num_cell_dofs))

        # FIXME: need communication in parallel here to figure out insert position for cell
        # connectivities and offsets
        assert mesh.comm.size == 1, "Parallel writing not implemented"

        # Offsets
        offsets = hdf.create_dataset("Offsets", (num_cells + 1,), dtype=np.int64)
        offsets[:] = geom_offsets
        geom_out = np.hstack(geometry_flattened)

        # Store global mesh connectivity for geometry
        top_set = hdf.create_dataset("Connectivity", (geom_out.size,), dtype=np.int64)
        top_set[:] = geom_out

        # Store cell types
        type_set = hdf.create_dataset("Types", (num_cells,), dtype=np.uint8)
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
        type_set[:] = cts

        # Geom dofmap offset
        con_part = hdf.create_dataset("NumberOfConnectivityIds", (1,), dtype=np.int64)
        con_part[0] = geom_offsets[-1]

        # Store meta-variable used by VTKHDF5 when partitioning data. We do not partition data,
        # as then everything becomes "local index", and cannot be read in again
        # in parallel on M processes.
        hdf.create_dataset(
            "NumberOfCells",
            (1,),
            dtype=np.int64,
            data=np.array([num_cells], dtype=np.int64),
        )

        # Similar metadata for number of nodes
        num_points = hdf.create_dataset("NumberOfPoints", (1,), dtype=np.int64)
        num_points[0] = geom_imap.size_global
