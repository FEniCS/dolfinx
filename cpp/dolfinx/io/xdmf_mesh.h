// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "xdmf_utils.h"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/mesh/MeshTags.h>
#include <hdf5.h>
#include <mpi.h>
#include <pugixml.hpp>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace pugi
{
class xml_node;
} // namespace pugi

namespace dolfinx
{

namespace mesh
{
template <std::floating_point T>
class Geometry;
template <std::floating_point T>
class Mesh;
class Topology;
} // namespace mesh

/// Low-level methods for reading XDMF files
namespace io::xdmf_mesh
{

/// Add Mesh to xml node
///
/// Creates new Grid with Topology and Geometry xml nodes for mesh. In
/// HDF file data is stored under path prefix.
template <std::floating_point U>
void add_mesh(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
              const mesh::Mesh<U>& mesh, const std::string& path_prefix);

/// Add Topology xml node
/// @param[in] comm
/// @param[in] xml_node
/// @param[in] h5_id
/// @param[in] path_prefix
/// @param[in] topology
/// @param[in] geometry
/// @param[in] cell_dim Dimension of mesh entities to save
/// @param[in] entities Local-to-process indices of mesh entities
/// whose topology will be saved. This is used to save subsets of Mesh.
template <std::floating_point U>
void add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                       std::string path_prefix, const mesh::Topology& topology,
                       const mesh::Geometry<U>& geometry, int cell_dim,
                       std::span<const std::int32_t> entities);

/// Add Geometry xml node
template <std::floating_point U>
void add_geometry_data(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                       std::string path_prefix,
                       const mesh::Geometry<U>& geometry);

/// @brief Read geometry (coordinate) data.
///
/// @returns The coordinates of each 'node'. The returned data is (0) an
/// array holding the coordinates (row-major storage) and (1) the shape
/// of the coordinate array. The shape is `(num_nodes, geometric
/// dimension)`.
std::pair<std::variant<std::vector<float>, std::vector<double>>,
          std::array<std::size_t, 2>>
read_geometry_data(MPI_Comm comm, hid_t h5_id, const pugi::xml_node& node);

/// @brief Read topology (cell connectivity) data.
///
/// @returns Mesh topology in DOLFINx ordering, where data row `i` lists
/// the 'nodes' of cell `i`. The returned data is (0) an array holding
/// the topology data (row-major storage) and (1) the shape of the
/// topology array. The shape is `(num_cells, num_nodes_per_cell)`
std::pair<std::vector<std::int64_t>, std::array<std::size_t, 2>>
read_topology_data(MPI_Comm comm, hid_t h5_id, const pugi::xml_node& node);

/// Add mesh tags to XDMF file
template <typename T, std::floating_point U>
void add_meshtags(MPI_Comm comm, const mesh::MeshTags<T>& meshtags,
                  const mesh::Geometry<U>& geometry, pugi::xml_node& xml_node,
                  hid_t h5_id, const std::string& name)
{
  spdlog::info("XDMF: add meshtags ({})", name.c_str());
  // Get mesh
  const int dim = meshtags.dim();
  std::shared_ptr<const common::IndexMap> entity_map
      = meshtags.topology()->index_map(dim);
  if (!entity_map)
  {
    throw std::runtime_error("Missing entities. Did you forget to call "
                             "dolfinx::mesh::Topology::create_entities?");
  }
  const std::int32_t num_local_entities = entity_map->size_local();

  // Find number of tagged entities in local range
  auto it = std::ranges::lower_bound(meshtags.indices(), num_local_entities);
  const int num_active_entities = std::distance(meshtags.indices().begin(), it);

  const std::string path_prefix = "/MeshTags/" + name;
  xdmf_mesh::add_topology_data(
      comm, xml_node, h5_id, path_prefix, *meshtags.topology(), geometry, dim,
      std::span<const std::int32_t>(meshtags.indices().data(),
                                    num_active_entities));

  // Add attribute node with values
  pugi::xml_node attribute_node = xml_node.append_child("Attribute");
  assert(attribute_node);
  attribute_node.append_attribute("Name") = name.c_str();
  attribute_node.append_attribute("AttributeType") = "Scalar";
  attribute_node.append_attribute("Center") = "Cell";

  std::int64_t global_num_values = 0;
  const std::int64_t local_num_values = num_active_entities;
  MPI_Allreduce(&local_num_values, &global_num_values, 1, MPI_INT64_T, MPI_SUM,
                comm);
  const std::int64_t num_local = num_active_entities;
  std::int64_t offset = 0;
  MPI_Exscan(&num_local, &offset, 1, MPI_INT64_T, MPI_SUM, comm);
  const bool use_mpi_io = (dolfinx::MPI::size(comm) > 1);
  xdmf_utils::add_data_item(
      attribute_node, h5_id, path_prefix + std::string("/Values"),
      std::span<const T>(meshtags.values().data(), num_active_entities), offset,
      {global_num_values, 1},
      std::string(io::xdmf_integral_float<T>::data_type), use_mpi_io);
}
} // namespace io::xdmf_mesh
} // namespace dolfinx
