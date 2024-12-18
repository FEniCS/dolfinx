from enum import Enum
from dolfinx.mesh import Mesh, entities_to_geometry
import dolfinx.cpp as _cpp
from pathlib import Path
import h5py
import numpy as np
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
        else:
            raise ValueError(f"Unknown cell type: {self}")

    def __int__(self) -> int:
        return self.value
    



def write(mesh: Mesh, filename: str | Path):
    """
    Write mesh to VTK HDF5 format
    """
    comm = mesh.comm
    fname = Path(filename).with_suffix(".vtkhdf")
    with h5py.File(fname, "w", driver="mpio", comm=comm) as inf:
        metadata = inf.create_group(np.bytes_("Metadata"))
        metadata.attrs["gdim"] = mesh.geometry.dim
        hdf = inf.create_group(np.bytes_("VTKHDF"))
        h5py.string_dtype(encoding="ascii")
        hdf.attrs["Version"] = [2, 2]
        hdf.attrs["Type"] = np.bytes_("UnstructuredGrid")

        geom_imap = mesh.geometry.index_map()
        gdim = mesh.geometry.dim
        geom_global_shape = (geom_imap.size_global, gdim)
        gdtype = mesh.geometry.x.dtype
        geom_irange = geom_imap.local_range
        
        p_string = np.bytes_("Points")
        geom_set = hdf.create_dataset(p_string, geom_global_shape, dtype=gdtype)
        geom_set[geom_irange[0]:geom_irange[1], :] = mesh.geometry.x[:geom_imap.size_local, :]


        #cell_map = mesh.topology.
        cell_maps = [imap for imap in mesh.topology.index_maps(mesh.topology.dim)]
        
        for i in range(len(cell_maps)):



        # Put global topology
        top_set = hdf.create_dataset(
            "Connectivity", (mesh.topology_offset[-1],), dtype=np.int64
        )
        top_set[:] = mesh.topology_array

        # Put cell type
        num_cells = len(mesh.topology_offset) - 1
        type_set = hdf.create_dataset("Types", (num_cells,), dtype=np.uint8)

        cts = np.asarray(
            [int(VTKCellType.from_value(ct)) for ct in mesh.cell_types], dtype=np.uint8
        )
        type_set[:] = cts

        # Geom dofmap offset
        con_part = hdf.create_dataset("NumberOfConnectivityIds", (1,), dtype=np.int64)
        con_part[0] = mesh.topology_offset[-1]

        # Num cells
        hdf.create_dataset(
            "NumberOfCells",
            (1,),
            dtype=np.int64,
            data=np.array([num_cells], dtype=np.int64),
        )

        # num points
        num_points = hdf.create_dataset("NumberOfPoints", (1,), dtype=np.int64)
        num_points[0] = mesh.geometry.shape[0]

        # Offsets
        offsets = hdf.create_dataset("Offsets", (num_cells + 1,), dtype=np.int64)
        offsets[:] = mesh.topology_offset

        # Add celldata
        if len(mesh.cell_values) > 0:
            cv = hdf.create_group("CellData")
            cv.attrs["Scalars"] = ["Cell_Markers"]
            cv.create_dataset("Cell_Markers", shape=(num_cells,), data=mesh.cell_values)

