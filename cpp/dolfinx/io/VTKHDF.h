// Copyright (C) 2024-2025 Chris Richardson, Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "HDF5Interface.h"

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/utils.h>

#include <map>
#include <vector>

namespace dolfinx::io::VTKHDF
{

/// @brief Write a mesh to VTKHDF format
/// @tparam U Scalar type of the mesh
/// @param filename File to write to
/// @param mesh Mesh
template <typename U>
void write_mesh(std::string filename, const mesh::Mesh<U>& mesh);

/// @brief  Read a mesh from VTKHDF format file
/// @tparam U Scalar type of mesh
/// @param comm MPI Communicator for reading mesh
/// @param filename Filename
/// @param gdim Geometric dimension
/// @return a mesh
template <typename U>
mesh::Mesh<U> read_mesh(MPI_Comm comm, std::string filename, std::size_t gdim = 3);

} // namespace dolfinx::io::VTKHDF

using namespace dolfinx;

template <typename U>
void io::VTKHDF::write_mesh(std::string filename, const mesh::Mesh<U>& mesh)
{
  hid_t h5file = io::hdf5::open_file(mesh.comm(), filename, "w", true);

  // Create VTKHDF group
  io::hdf5::add_group(h5file, "VTKHDF");

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
  auto cell_types = mesh.topology()->entity_types(mesh.topology()->dim());

  std::vector cell_index_maps
      = mesh.topology()->index_maps(mesh.topology()->dim());
  std::vector<std::int32_t> num_cells;
  std::vector<std::int64_t> num_cells_global;
  for (auto im : cell_index_maps)
  {
    num_cells.push_back(im->size_local());
    num_cells_global.push_back(im->size_global());
  }

  // Geometry dofmap and points
  auto geom_imap = mesh.geometry().index_map();
  std::int64_t size_global = geom_imap->size_global();
  std::vector<std::int64_t> geom_global_shape = {size_global, 3};
  auto geom_irange = geom_imap->local_range();

  io::hdf5::write_dataset(h5file, "/VTKHDF/Points", mesh.geometry().x().data(),
                          geom_irange, geom_global_shape, true, false);

  io::hdf5::write_dataset(h5file, "VTKHDF/NumberOfPoints", &size_global, {0, 1},
                          {1}, true, false);

  // VTKHDF stores the cells as an adjacency list,                                \
  //        where cell types might be jumbled up.
  std::vector<std::int64_t> topology_flattened;
  std::vector<std::int64_t> topology_offsets;
  std::vector<std::uint8_t> vtkcelltypes;
  for (int i = 0; i < cell_index_maps.size(); ++i)
  {
    auto g_dofmap = mesh.geometry().dofmap(i);

    std::vector<std::int32_t> local_dm;
    local_dm.reserve(g_dofmap.extent(1) * num_cells[i]);
    auto perm = io::cells::perm_vtk(cell_types[i], g_dofmap.extent(1));
    auto inverse_perm = io::cells::transpose(perm);
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
        io::cells::get_vtk_cell_type(cell_types[i], mesh.topology()->dim()));
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
    auto r = cell_index_maps[i]->local_range();
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

  std::for_each(topology_offsets.begin(), topology_offsets.end(),
                [topology_start](std::int64_t& t) { t += topology_start; });

  std::int64_t num_all_cells_global
      = std::accumulate(num_cells_global.begin(), num_cells_global.end(), 0);
  io::hdf5::write_dataset(h5file, "/VTKHDF/Offsets", topology_offsets.data(),
                          {offset_start_position + 1, offset_stop_position + 1},
                          {num_all_cells_global + 1}, true, false);

  // Store global mesh connectivity
  std::int64_t topology_size_global
      = std::inner_product(num_nodes_per_cell.begin(), num_nodes_per_cell.end(),
                           num_cells_global.begin(), 0);

  std::int64_t topology_stop = topology_start + topology_flattened.size();
  io::hdf5::write_dataset(
      h5file, "/VTKHDF/Connectivity", topology_flattened.data(),
      {topology_start, topology_stop}, {topology_size_global}, true, false);

  // Store cell types
  io::hdf5::write_dataset(h5file, "/VTKHDF/Types", vtkcelltypes.data(),
                          {offset_start_position, offset_stop_position},
                          {num_all_cells_global}, true, false);

  io::hdf5::write_dataset(h5file, "/VTKHDF/NumberOfConnectivityIds",
                          &topology_size_global, {0, 1}, {1}, true, false);

  io::hdf5::write_dataset(h5file, "/VTKHDF/NumberOfCells",
                          &num_all_cells_global, {0, 1}, {1}, true, false);

  io::hdf5::close_file(h5file);
}

template <typename U>
mesh::Mesh<U> io::VTKHDF ::read_mesh(MPI_Comm comm, std::string filename,
                                     std::size_t gdim = 3)
{
  hid_t h5file = io::hdf5::open_file(comm, filename, "r", true);

  std::vector<std::int64_t> shape
      = io::hdf5::get_dataset_shape(h5file, "/VTKHDF/Types");
  int rank = MPI::rank(comm);
  int size = MPI::size(comm);
  auto local_cell_range = dolfinx::MPI::local_range(rank, shape[0], size);

  hid_t dset_id = io::hdf5::open_dataset(h5file, "/VTKHDF/Types");
  std::vector<std::uint8_t> types
      = io::hdf5::read_dataset<std::uint8_t>(dset_id, local_cell_range, true);
  H5Dclose(dset_id);
  std::vector<std::array<std::uint8_t, 2>> types_unique;
  types_unique.reserve(types.size());

  // Create reverse map from VTK to dolfinx cell types
  std::map<std::uint8_t, mesh::CellType> vtk_to_dolfinx;
  constexpr std::array<mesh::CellType, 8> dolfinx_cells
      = {mesh::CellType::point,       mesh::CellType::interval,
         mesh::CellType::triangle,    mesh::CellType::quadrilateral,
         mesh::CellType::tetrahedron, mesh::CellType::prism,
         mesh::CellType::pyramid,     mesh::CellType::hexahedron};
  for (auto dolfinx_type : dolfinx_cells)
  {
    vtk_to_dolfinx.insert({io::cells::get_vtk_cell_type(
                               dolfinx_type, mesh::cell_dim(dolfinx_type)),
                           dolfinx_type});
  }

  // Read in offsets to determine the different cell-types in the mesh
  dset_id = io::hdf5::open_dataset(h5file, "/VTKHDF/Offsets");
  std::vector<std::int64_t> offsets = io::hdf5::read_dataset<std::int64_t>(
      dset_id, {local_cell_range[0], local_cell_range[1] + 1}, true);
  H5Dclose(dset_id);

  // Convert cell offsets to cell type and cell degree tuples
  std::vector<std::uint8_t> cell_degrees;
  cell_degrees.reserve(types.size());
  for (std::size_t i = 0; i < types.size(); ++i)
  {
    std::uint8_t num_nodes = offsets[i + 1] - offsets[i];
    std::uint8_t cell_degree
        = io::cells::cell_degree(vtk_to_dolfinx.at(types[i]), num_nodes);
    types_unique.push_back({types[i], cell_degree});
    cell_degrees.push_back(cell_degree);
  }
  {
    std::ranges::sort(types_unique);
    auto [unique_end, range_end] = std::ranges::unique(types_unique);
    types_unique.erase(unique_end, range_end);
  }

  // Share cell types with all processes to make global list of cell types
  // FIXME: amount of data is small, but number of connections does not scale.
  std::int32_t count = 2 * types_unique.size();
  std::vector<std::int32_t> recv_count(size);
  std::vector<std::int32_t> recv_offsets(size + 1, 0);
  MPI_Allgather(&count, 1, MPI_INT32_T, recv_count.data(), 1, MPI_INT32_T,
                comm);
  std::partial_sum(recv_count.begin(), recv_count.end(),
                   recv_offsets.begin() + 1);

  std::vector<std::array<std::uint8_t, 2>> recv_types;
  recv_types.reserve(recv_offsets.back());
  {
    std::vector<std::uint8_t> send_types;
    send_types.reserve(count);
    std::for_each(types_unique.begin(), types_unique.end(),
                  [&send_types](const auto& t)
                  {
                    send_types.push_back(t.front());
                    send_types.push_back(t.back());
                  });

    std::vector<std::uint8_t> recv_types_buffer(recv_offsets.back());
    MPI_Allgatherv(send_types.data(), send_types.size(), MPI_UINT8_T,
                   recv_types_buffer.data(), recv_count.data(),
                   recv_offsets.data(), MPI_UINT8_T, comm);
    md::mdspan<const std::uint8_t,
               md::extents<std::size_t, md::dynamic_extent, 2>>
        recv_types_span(recv_types_buffer.data(), recv_types_buffer.size() / 2,
                        2);
    {
      for (std::size_t i = 0; i < recv_types_span.extent(0); ++i)
        recv_types.push_back({recv_types_span(i, 0), recv_types_span(i, 1)});

      std::ranges::sort(recv_types);
      auto [unique_end, range_end] = std::ranges::unique(recv_types);
      recv_types.erase(unique_end, range_end);
    }
  }

  // Map from VTKCellType to index in list of (cell types, degree)
  std::map<std::array<std::uint8_t, 2>, std::int32_t> type_to_index;
  std::vector<mesh::CellType> dolfinx_cell_type;
  std::vector<std::uint8_t> dolfinx_cell_degree;
  std::vector<std::uint32_t> num_nodes_per_cell;
  dolfinx_cell_type.reserve(recv_types.size());
  dolfinx_cell_degree.reserve(recv_types.size());
  std::size_t c = 0;
  std::for_each(recv_types.begin(), recv_types.end(),
                [&type_to_index, &dolfinx_cell_type, &dolfinx_cell_degree,
                 &vtk_to_dolfinx, &c](const auto& ct)
                {
                  mesh::CellType cell_type = vtk_to_dolfinx.at(ct.front());
                  type_to_index.insert({ct, c++});
                  dolfinx_cell_degree.push_back(ct.back());
                  dolfinx_cell_type.push_back(cell_type);
                });

  dset_id = io::hdf5::open_dataset(h5file, "/VTKHDF/NumberOfPoints");
  std::vector<std::int64_t> npoints
      = io::hdf5::read_dataset<std::int64_t>(dset_id, {0, 1}, true);
  H5Dclose(dset_id);
  spdlog::info("Mesh with {} points", npoints[0]);
  auto local_point_range = MPI::local_range(rank, npoints[0], size);

  std::vector<std::int64_t> x_shape
      = io::hdf5::get_dataset_shape(h5file, "/VTKHDF/Points");
  dset_id = io::hdf5::open_dataset(h5file, "/VTKHDF/Points");
  std::vector<U> points_local
      = io::hdf5::read_dataset<U>(dset_id, local_point_range, true);
  H5Dclose(dset_id);

  // Remove coordinates if gdim!=3
  assert(gdim < 4);
  std::vector<U> points_pruned((local_point_range[1] - local_point_range[0])
                               * gdim);
  for (std::size_t i = 0; i < local_point_range[1] - local_point_range[0]; ++i)
  {
    std::copy_n(points_local.begin() + i * 3, gdim,
                points_pruned.begin() + i * gdim);
  }

  dset_id = io::hdf5::open_dataset(h5file, "/VTKHDF/Connectivity");
  std::vector<std::int64_t> topology = io::hdf5::read_dataset<std::int64_t>(
      dset_id, {offsets.front(), offsets.back()}, true);
  H5Dclose(dset_id);
  const std::int64_t offset0 = offsets.front();
  std::for_each(offsets.begin(), offsets.end(),
                [&offset0](std::int64_t& v) { v -= offset0; });
  io::hdf5::close_file(h5file);

  // Create cell topologies for each celltype in mesh
  std::vector<std::vector<std::int64_t>> cells_local(recv_types.size());
  for (std::size_t j = 0; j < types.size(); ++j)
  {
    std::int32_t type_index = type_to_index.at({types[j], cell_degrees[j]});
    mesh::CellType cell_type = dolfinx_cell_type[type_index];
    auto perm = io::cells::perm_vtk(cell_type, offsets[j + 1] - offsets[j]);
    for (std::size_t k = 0; k < offsets[j + 1] - offsets[j]; ++k)
      cells_local[type_index].push_back(topology[perm[k] + offsets[j]]);
  }

  // Make coordinate elements
  std::vector<fem::CoordinateElement<U>> coordinate_elements;
  std::transform(
      dolfinx_cell_type.cbegin(), dolfinx_cell_type.cend(),
      dolfinx_cell_degree.cbegin(), std::back_inserter(coordinate_elements),
      [](const auto& cell_type, const auto& cell_degree)
      {
        basix::element::lagrange_variant variant
            = basix::element::lagrange_variant::unset;
        if (cell_degree > 2)
          variant = basix::element::lagrange_variant::equispaced;
        return fem::CoordinateElement<U>(cell_type, cell_degree, variant);
      });

  auto part = create_cell_partitioner(mesh::GhostMode::none);
  std::vector<std::span<const std::int64_t>> cells_span;
  for (auto& cells : cells_local)
    cells_span.push_back(cells);
  std::array<std::size_t, 2> xs = {(std::size_t)x_shape[0], (std::size_t)gdim};
  return create_mesh(comm, comm, cells_span, coordinate_elements, comm,
                     points_pruned, xs, part);
}
