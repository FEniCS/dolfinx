
#include "VTKHDFFile.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>

using namespace dolfinx;

template <typename U>
void io::VTKHDF::write_mesh(std::string filename, const mesh::Mesh<U>& mesh)
{
  hid_t h5file = io::hdf5::open_file(mesh.comm(), filename, "w", true);

  // Create VTKHDF group
  io::hdf5::add_group(h5file, "VTKHDF");

  hid_t vtk_group = H5Gopen(h5file, "VTKHDF", H5P_DEFAULT);

  hsize_t dims = 2;
  hid_t space_id = H5Screate_simple(1, &dims, NULL);
  hid_t attr_id = H5Acreate(vtk_group, "Version", H5T_NATIVE_INT32, space_id,
                            H5P_DEFAULT, H5P_DEFAULT);
  std::array<std::int32_t, 2> version = {2, 2};
  H5Awrite(attr_id, H5T_NATIVE_INT32, version.data());
  H5Aclose(attr_id);
  H5Sclose(space_id);

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
  std::int32_t gdim = mesh.geometry().dim();
  std::int64_t size_global = geom_imap->size_global();
  std::vector<std::int64_t> geom_global_shape = {size_global, gdim};
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
    for (int j = 0; j < num_cells[i]; ++j)
      for (int k = 0; k < g_dofmap.extent(1); ++k)
        local_dm.push_back(g_dofmap(j, perm[k]));

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
    num_nodes_per_cell.push_back(mesh.geometry().cmap(i).dim());
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
}

template void io::VTKHDF::write_mesh(std::string filename,
                                     const mesh::Mesh<double>& mesh);
