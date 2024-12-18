# SPDX license: MIT
# Author: Jørgen S. Dokken
from enum import Enum
from pathlib import Path

from mpi4py import MPI

import h5py
import numpy as np

import basix.ufl
import dolfinx
from dolfinx.cpp.io import perm_vtk
from dolfinx.cpp.mesh import create_cell_partitioner, create_mesh
from dolfinx.fem import coordinate_element
from dolfinx.mesh import GhostMode


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
            return "Polyline"
        elif self == VTKCellType.triangle:
            return "triangle"
        elif self == VTKCellType.quadrilateral:
            return "quadrilateral"
        elif self == VTKCellType.tetrahedron:
            return "tetrahedron"
        elif self == VTKCellType.hexahedron:
            return "hexahedron"
        elif self == VTKCellType.vertex:
            return "Polyvertex"
        else:
            raise ValueError(f"Unknown cell type: {self}")

    def __int__(self) -> int:
        return self.value


def cell_degree(ct: dolfinx.mesh.CellType, num_nodes: int):
    if ct == dolfinx.mesh.CellType.point:
        return 1
    elif ct == dolfinx.mesh.CellType.interval:
        return num_nodes - 1
    elif ct == dolfinx.mesh.CellType.triangle:
        n = (np.sqrt(1 + 8 * num_nodes) - 1) / 2
        if 2 * num_nodes != n * (n + 1):
            raise ValueError(f"Unknown triangle layout. Number of nodes: {num_nodes}")
        return n - 1
    elif ct == dolfinx.mesh.CellType.tetrahedron:
        n = 0
        while n * (n + 1) * (n + 2) < 6 * num_nodes:
            n += 1
        if n * (n + 1) * (n + 2) != 6 * num_nodes:
            raise ValueError(f"Unknown tetrahedron layout. Number of nodes: {num_nodes}")
        return n - 1

    elif ct == dolfinx.mesh.CellType.quadrilateral:
        n = np.sqrt(num_nodes)
        if num_nodes != n * n:
            raise ValueError(f"Unknown quadrilateral layout. Number of nodes: {num_nodes}")
        return n - 1
    elif ct == dolfinx.mesh.CellType.hexahedron:
        n = np.cbrt(num_nodes)
        if num_nodes != n * n * n:
            raise ValueError(f"Unknown hexahedron layout. Number of nodes: {num_nodes}")
        return n - 1
    elif ct == dolfinx.mesh.CellType.prism:
        if num_nodes == 6:
            return 1
        elif num_nodes == 15:
            return 2
        else:
            raise ValueError(f"Unknown prism layout. Number of nodes: {num_nodes}")
    elif ct == dolfinx.mesh.CellType.pyramid:
        if num_nodes == 5:
            return 1
        elif num_nodes == 13:
            return 2
        else:
            raise ValueError(f"Unknown pyramid layout. Number of nodes: {num_nodes}")
    else:
        raise ValueError(f"Unknown cell type {ct} with {num_nodes=}.")


def compute_local_range(comm: MPI.Intracomm, N: np.int64):
    """
    Divide a set of `N` objects into `M` partitions, where `M` is
    the size of the MPI communicator `comm`.

    NOTE: If N is not divisible by the number of ranks, the first `r`
    processes gets an extra value

    Returns the local range of values
    """
    rank = comm.rank
    size = comm.size
    n = N // size
    r = N % size
    # First r processes has one extra value
    if rank < r:
        return [rank * (n + 1), (rank + 1) * (n + 1)]
    else:
        return [rank * n + r, (rank + 1) * n + r]


def read_vtkhdf(
    filename: str | Path, comm: MPI.Intracomm
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read VTK HDF5 file and return the local piece of the geometry,
    topology, topology offsets and cell types
    """
    comm = MPI.COMM_WORLD
    filename = "test"
    fname = Path(filename).with_suffix(".vtkhdf")
    inf = h5py.File(fname, "r", driver="mpio", comm=comm)

    hdf = inf["VTKHDF"]
    num_cells_global = hdf["Types"].size

    local_cell_range = compute_local_range(comm, num_cells_global)
    cell_types_local = hdf["Types"][local_cell_range[0] : local_cell_range[1]]

    num_points_global = hdf["NumberOfPoints"][0]
    local_point_range = compute_local_range(comm, num_points_global)
    points_local = hdf["Points"][local_point_range[0] : local_point_range[1]]

    # Connectivity read
    offsets = hdf["Offsets"]
    local_connectivity_offset = offsets[local_cell_range[0] : local_cell_range[1] + 1]
    topology = hdf["Connectivity"][local_connectivity_offset[0] : local_connectivity_offset[-1]]
    inf.close()
    offset = local_connectivity_offset - local_connectivity_offset[0]

    return points_local, topology, offset, cell_types_local


def find_all_unique_cell_types(comm, cell_types, num_nodes):
    """
    Given a set of cell types and number of nodes per cell, find all unique cell types
    across all ranks.

    Args;
        comm: MPI communicator
        cell_types: Local cell types
        num_nodes: Number of nodes per cell
    """
    # Combine cell_types, num_nodes as tuple
    c_hash = np.zeros((2, len(cell_types)), dtype=np.int32)
    c_hash[0] = cell_types
    c_hash[1] = num_nodes
    indexes = np.unique(c_hash.T, axis=0, return_index=True)[1]
    local_unique_cells = c_hash.T[indexes]

    all_cell_types = np.vstack(comm.allgather(local_unique_cells))
    indexes = np.unique(all_cell_types, axis=0, return_index=True)[1]
    all_unique_cell_types = all_cell_types[indexes]
    return all_unique_cell_types


x, connectivity, offsets, cell_types = read_vtkhdf("test", MPI.COMM_WORLD)

num_nodes_per_cell = offsets[1:] - offsets[:-1]
unique_cells = find_all_unique_cell_types(MPI.COMM_WORLD, cell_types, num_nodes_per_cell)

# Compute mask for extracting connectivities for each unique cell type
masks = [
    np.flatnonzero((cell_types == ct) & (num_nodes_per_cell == nn)) for (ct, nn) in unique_cells
]


cell_types: list[basix.ufl._BasixElement] = []
connectivities: list[np.ndarray] = []
for cell_type, mask in zip(unique_cells, masks):
    d_ct = dolfinx.mesh.to_type(str(VTKCellType(cell_type[0])))
    degree = int(cell_degree(d_ct, cell_type[1]))
    permutation = perm_vtk(d_ct, cell_type[1])
    sub_connectivity = np.zeros((len(mask), cell_type[1]), dtype=connectivity.dtype)
    cell_types.append(
        coordinate_element(
            basix.ufl.element(
                "Lagrange",
                dolfinx.mesh.to_string(d_ct),
                degree,
                shape=(x.shape[1],),
            ).basix_element
        )._cpp_object
    )
    for i, cell in enumerate(mask):
        sub_connectivity[i][permutation] = connectivity[offsets[mask[i]] : offsets[mask[i] + 1]]
    connectivities.append(sub_connectivity.reshape(-1))


part = create_cell_partitioner(GhostMode.none)
mesh = create_mesh(
    MPI.COMM_WORLD,
    connectivities,
    cell_types,
    x,
    part,
)

print(mesh.topology.entity_types)
print(mesh.topology.connectivity((2, 0), (0, 0)))
print(mesh.topology.connectivity((2, 1), (0, 0)))
