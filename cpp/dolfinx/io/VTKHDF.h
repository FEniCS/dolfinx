// Copyright (C) 2024-2025 Chris Richardson, Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "HDF5Interface.h"
#include <algorithm>
#include <concepts>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/utils.h>
#include <map>
#include <vector>

namespace dolfinx::io::VTKHDF
{
/// @brief Write a mesh to VTKHDF format.
///
/// The mesh is written on the MPI communicator of the mesh.
///
/// @tparam U Scalar type of the mesh
/// @param filename Name of file to write to.
/// @param mesh Mesh to write to file.
template <std::floating_point U>
void write_mesh(std::string filename, const mesh::Mesh<U>& mesh)
{
  hid_t h5file = hdf5::open_file(mesh.comm(), filename, "w", true);

  // Create VTKHDF group
  hdf5::add_group(h5file, "VTKHDF");
  hid_t vtk_group = H5Gopen(h5file, "VTKHDF", H5P_DEFAULT);

  // Create "Version" attribute
  hsize_t dims = 2;
  hid_t space_id = H5Screate_simple(1, &dims, NULL);
  hid_t attr_id = H5Acreate(vtk_group, "Version", H5T_NATIVE_INT32, space_id,
                            H5P_DEFAULT, H5P_DEFAULT);
  std::array<std::int32_t, 2> version = {2, 2};
  H5Awrite(attr_id, H5T_NATIVE_INT32, version.data());
  H5Aclose(attr_id);
  H5Sclose(space_id);

  // Create "Type" attribute
  space_id = H5Screate(H5S_SCALAR);
  hid_t atype = H5Tcopy(H5T_C_S1);
  H5Tset_size(atype, 16);
  H5Tset_strpad(atype, H5T_STR_NULLTERM);
  attr_id
      = H5Acreate(vtk_group, "Type", atype, space_id, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id, atype, "UnstructuredGrid");
  H5Aclose(attr_id);
  H5Sclose(space_id);
  H5Gclose(vtk_group);

  // Extract topology information for each cell type
  std::vector<mesh::CellType> cell_types
      = mesh.topology()->entity_types(mesh.topology()->dim());

  std::vector cell_index_maps
      = mesh.topology()->index_maps(mesh.topology()->dim());
  std::vector<std::int32_t> num_cells;
  std::vector<std::int64_t> num_cells_global;
  for (auto& im : cell_index_maps)
  {
    num_cells.push_back(im->size_local());
    num_cells_global.push_back(im->size_global());
  }

  // Geometry dofmap and points
  std::shared_ptr<const common::IndexMap> geom_imap
      = mesh.geometry().index_map();
  std::int64_t size_global = geom_imap->size_global();
  std::vector<std::int64_t> geom_global_shape = {size_global, 3};
  std::array<std::int64_t, 2> geom_irange = geom_imap->local_range();
  hdf5::write_dataset(h5file, "/VTKHDF/Points", mesh.geometry().x().data(),
                      geom_irange, geom_global_shape, true, false);
  hdf5::write_dataset(h5file, "VTKHDF/NumberOfPoints", &size_global, {0, 1},
                      {1}, true, false);

  // Note: VTKHDF stores the cells as an adjacency list, where cell
  // types might be jumbled up
  std::vector<std::int64_t> topology_flattened;
  std::vector<std::int64_t> topology_offsets;
  std::vector<std::uint8_t> vtkcelltypes;
  for (int i = 0; i < cell_index_maps.size(); ++i)
  {
    md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>> g_dofmap
        = mesh.geometry().dofmap(i);

    std::vector<std::uint16_t> perm
        = cells::perm_vtk(cell_types[i], g_dofmap.extent(1));
    std::vector<std::uint16_t> inverse_perm = cells::transpose(perm);
    std::vector<std::int32_t> local_dm;
    local_dm.reserve(g_dofmap.extent(1) * num_cells[i]);
    for (int j = 0; j < num_cells[i]; ++j)
      for (int k = 0; k < g_dofmap.extent(1); ++k)
        local_dm.push_back(g_dofmap(j, inverse_perm[k]));

    std::vector<std::int64_t> global_dm(local_dm.size());
    geom_imap->local_to_global(local_dm, global_dm);

    topology_flattened.insert(topology_flattened.end(), global_dm.begin(),
                              global_dm.end());
    topology_offsets.insert(topology_offsets.end(), g_dofmap.extent(0),
                            g_dofmap.extent(1));
    vtkcelltypes.insert(
        vtkcelltypes.end(), cell_index_maps[i]->size_local(),
        cells::get_vtk_cell_type(cell_types[i], mesh.topology()->dim()));
  }

  // Create topo_offsets
  std::partial_sum(topology_offsets.cbegin(), topology_offsets.cend(),
                   topology_offsets.begin());

  std::vector<int> num_nodes_per_cell;
  std::vector<std::int64_t> cell_start_pos;
  std::vector<std::int64_t> cell_stop_pos;
  for (int i = 0; i < cell_index_maps.size(); ++i)
  {
    num_nodes_per_cell.push_back(mesh.geometry().cmaps()[i].dim());
    std::array<std::int64_t, 2> r = cell_index_maps[i]->local_range();
    cell_start_pos.push_back(r[0]);
    cell_stop_pos.push_back(r[1]);
  }

  // Compute overall cell offset from offsets for each cell type
  std::int64_t offset_start_position
      = std::accumulate(cell_start_pos.begin(), cell_start_pos.end(), 0);
  std::int64_t offset_stop_position
      = std::accumulate(cell_stop_pos.begin(), cell_stop_pos.end(), 0);

  // Compute overall topology offset from offsets for each cell type
  std::int64_t topology_start
      = std::inner_product(num_nodes_per_cell.begin(), num_nodes_per_cell.end(),
                           cell_start_pos.begin(), 0);

  std::transform(topology_offsets.cbegin(), topology_offsets.cend(),
                 topology_offsets.begin(),
                 [topology_start](auto x) { return x + topology_start; });

  std::int64_t num_all_cells_global
      = std::accumulate(num_cells_global.begin(), num_cells_global.end(), 0);
  hdf5::write_dataset(h5file, "/VTKHDF/Offsets", topology_offsets.data(),
                      {offset_start_position + 1, offset_stop_position + 1},
                      {num_all_cells_global + 1}, true, false);

  // Store global mesh connectivity
  std::int64_t topology_size_global
      = std::inner_product(num_nodes_per_cell.begin(), num_nodes_per_cell.end(),
                           num_cells_global.begin(), 0);

  std::int64_t topology_stop = topology_start + topology_flattened.size();
  hdf5::write_dataset(h5file, "/VTKHDF/Connectivity", topology_flattened.data(),
                      {topology_start, topology_stop}, {topology_size_global},
                      true, false);

  // Store cell types
  hdf5::write_dataset(h5file, "/VTKHDF/Types", vtkcelltypes.data(),
                      {offset_start_position, offset_stop_position},
                      {num_all_cells_global}, true, false);
  hdf5::write_dataset(h5file, "/VTKHDF/NumberOfConnectivityIds",
                      &topology_size_global, {0, 1}, {1}, true, false);
  hdf5::write_dataset(h5file, "/VTKHDF/NumberOfCells", &num_all_cells_global,
                      {0, 1}, {1}, true, false);
  hdf5::close_file(h5file);
}

/// @brief Read a mesh from a VTKHDF format file.
///
/// @tparam U Scalar type of mesh
/// @param comm MPI Communicator for reading mesh
/// @param filename Name of the file to read from.
/// @param gdim Geometric dimension of the mesh. All VTK meshes are
/// embedded in 3D. Use this argument for meshes that should be in 1D or
/// 2D.
/// @return The mesh read from file.
template <std::floating_point U>
mesh::Mesh<U> read_mesh(MPI_Comm comm, std::string filename,
                        std::size_t gdim = 3)
{
  hid_t h5file = hdf5::open_file(comm, filename, "r", true);

  std::vector<std::int64_t> shape
      = hdf5::get_dataset_shape(h5file, "/VTKHDF/Types");
  int rank = dolfinx::MPI::rank(comm);
  int mpi_size = dolfinx::MPI::size(comm);
  std::array<std::int64_t, 2> local_cell_range
      = dolfinx::MPI::local_range(rank, shape[0], mpi_size);

  hid_t dset_id = hdf5::open_dataset(h5file, "/VTKHDF/Types");
  std::vector<std::uint8_t> types
      = hdf5::read_dataset<std::uint8_t>(dset_id, local_cell_range, true);
  H5Dclose(dset_id);

  // Create reverse map (VTK -> DOLFINx cell type)
  std::map<std::uint8_t, mesh::CellType> vtk_to_dolfinx;
  {
    for (auto type : {mesh::CellType::point, mesh::CellType::interval,
                      mesh::CellType::triangle, mesh::CellType::quadrilateral,
                      mesh::CellType::tetrahedron, mesh::CellType::prism,
                      mesh::CellType::pyramid, mesh::CellType::hexahedron})
    {
      vtk_to_dolfinx.insert(
          {cells::get_vtk_cell_type(type, mesh::cell_dim(type)), type});
    }
  }

  // Read in offsets to determine the different cell-types in the mesh
  dset_id = hdf5::open_dataset(h5file, "/VTKHDF/Offsets");
  std::vector<std::int64_t> offsets = hdf5::read_dataset<std::int64_t>(
      dset_id, {local_cell_range[0], local_cell_range[1] + 1}, true);
  H5Dclose(dset_id);

  // Convert cell offsets to cell type and cell degree tuples
  std::vector<std::array<std::uint8_t, 2>> types_unique;
  std::vector<std::uint8_t> cell_degrees;
  for (std::size_t i = 0; i < types.size(); ++i)
  {
    std::int64_t num_nodes = offsets[i + 1] - offsets[i];
    int cell_degree
        = cells::cell_degree(vtk_to_dolfinx.at(types[i]), num_nodes);
    types_unique.push_back({types[i], cell_degree});
    cell_degrees.push_back(cell_degree);
  }
  {
    std::ranges::sort(types_unique);
    auto [unique_end, range_end] = std::ranges::unique(types_unique);
    types_unique.erase(unique_end, range_end);
  }

  // Share cell types with all processes to make global list of cell
  // types
  // FIXME: amount of data is small, but number of connections does not
  // scale
  int count = 2 * types_unique.size();
  std::vector<std::int32_t> recv_count(mpi_size);
  MPI_Allgather(&count, 1, MPI_INT32_T, recv_count.data(), 1, MPI_INT32_T,
                comm);
  std::vector<std::int32_t> recv_offsets(mpi_size + 1, 0);
  std::partial_sum(recv_count.begin(), recv_count.end(),
                   recv_offsets.begin() + 1);

  std::vector<std::array<std::uint8_t, 2>> recv_types;
  {
    std::vector<std::uint8_t> send_types;
    for (std::array<std::uint8_t, 2> t : types_unique)
      send_types.insert(send_types.end(), t.begin(), t.end());

    std::vector<std::uint8_t> recv_types_buffer(recv_offsets.back());
    MPI_Allgatherv(send_types.data(), send_types.size(), MPI_UINT8_T,
                   recv_types_buffer.data(), recv_count.data(),
                   recv_offsets.data(), MPI_UINT8_T, comm);

    for (std::size_t i = 0; i < recv_types_buffer.size(); i += 2)
      recv_types.push_back({recv_types_buffer[i], recv_types_buffer[i + 1]});

    std::ranges::sort(recv_types);
    auto [unique_end, range_end] = std::ranges::unique(recv_types);
    recv_types.erase(unique_end, range_end);
  }

  // Map from VTKCellType to index in list of (cell types, degree)
  std::map<std::array<std::uint8_t, 2>, std::int32_t> type_to_index;
  std::vector<mesh::CellType> dolfinx_cell_type;
  std::vector<std::uint8_t> dolfinx_cell_degree;
  for (std::array<std::uint8_t, 2> ct : recv_types)
  {
    mesh::CellType cell_type = vtk_to_dolfinx.at(ct[0]);
    type_to_index.insert({ct, dolfinx_cell_degree.size()});
    dolfinx_cell_degree.push_back(ct[1]);
    dolfinx_cell_type.push_back(cell_type);
  }

  dset_id = hdf5::open_dataset(h5file, "/VTKHDF/NumberOfPoints");
  std::vector npoints = hdf5::read_dataset<std::int64_t>(dset_id, {0, 1}, true);
  H5Dclose(dset_id);
  spdlog::info("Mesh with {} points", npoints[0]);
  std::array<std::int64_t, 2> local_point_range
      = dolfinx::MPI::local_range(rank, npoints[0], mpi_size);

  std::vector<std::int64_t> x_shape
      = hdf5::get_dataset_shape(h5file, "/VTKHDF/Points");
  dset_id = hdf5::open_dataset(h5file, "/VTKHDF/Points");
  std::vector<U> points_local
      = hdf5::read_dataset<U>(dset_id, local_point_range, true);
  H5Dclose(dset_id);

  // Remove coordinates if gdim != 3
  assert(gdim <= 3);
  std::vector<U> points_pruned((local_point_range[1] - local_point_range[0])
                               * gdim);
  for (std::size_t i = 0; i < local_point_range[1] - local_point_range[0]; ++i)
  {
    std::copy_n(points_local.begin() + i * 3, gdim,
                points_pruned.begin() + i * gdim);
  }

  dset_id = hdf5::open_dataset(h5file, "/VTKHDF/Connectivity");
  std::vector<std::int64_t> topology = hdf5::read_dataset<std::int64_t>(
      dset_id, {offsets.front(), offsets.back()}, true);
  H5Dclose(dset_id);
  std::transform(offsets.cbegin(), offsets.cend(), offsets.begin(),
                 [offset = offsets.front()](auto x) { return x - offset; });
  hdf5::close_file(h5file);

  // Create cell topologies for each celltype in mesh
  std::vector<std::vector<std::int64_t>> cells_local(recv_types.size());
  for (std::size_t j = 0; j < types.size(); ++j)
  {
    std::int32_t type_index = type_to_index.at({types[j], cell_degrees[j]});
    mesh::CellType cell_type = dolfinx_cell_type[type_index];
    std::vector<std::uint16_t> perm
        = cells::perm_vtk(cell_type, offsets[j + 1] - offsets[j]);
    for (std::size_t k = 0; k < offsets[j + 1] - offsets[j]; ++k)
      cells_local[type_index].push_back(topology[perm[k] + offsets[j]]);
  }

  // Make coordinate elements
  std::vector<fem::CoordinateElement<U>> coordinate_elements;
  std::transform(
      dolfinx_cell_type.cbegin(), dolfinx_cell_type.cend(),
      dolfinx_cell_degree.cbegin(), std::back_inserter(coordinate_elements),
      [](auto cell_type, auto cell_degree)
      {
        basix::element::lagrange_variant variant
            = (cell_degree > 2) ? basix::element::lagrange_variant::equispaced
                                : basix::element::lagrange_variant::unset;
        return fem::CoordinateElement<U>(cell_type, cell_degree, variant);
      });

  auto part = create_cell_partitioner(mesh::GhostMode::none);
  std::vector<std::span<const std::int64_t>> cells_span(cells_local.begin(),
                                                        cells_local.end());
  return mesh::create_mesh(comm, comm, cells_span, coordinate_elements, comm,
                           points_pruned, {x_shape[0], gdim},
                           part);
}
} // namespace dolfinx::io::VTKHDF
