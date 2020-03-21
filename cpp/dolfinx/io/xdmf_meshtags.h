// Copyright (C) 2020 Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "HDF5Interface.h"
#include "xdmf_mesh.h"
#include "xdmf_utils.h"
#include <dolfinx/mesh/MeshTags.h>
#include "pugixml.hpp"


namespace dolfinx
{

namespace io
{
namespace xdmf_meshtags
{

template <typename T>
void add_meshtags(MPI_Comm comm, const mesh::MeshTags<T>& meshtags,
                  pugi::xml_node& xml_node, const hid_t h5_id,
                  const std::string name)
{
  // Get mesh
  assert(meshtags.mesh());
  std::shared_ptr<const mesh::Mesh> mesh = meshtags.mesh();
  const int dim = meshtags.dim();

  // Prepare an array of active entities
  auto map_e = mesh->topology().index_map(dim);
  assert(map_e);
  std::vector<std::int32_t> active_entities(meshtags.indices().data(),
                                            meshtags.indices().data()
                                                + meshtags.indices().rows());

  const std::string path_prefix = "/MeshTags/" + name;
  xdmf_mesh::add_topology_data(comm, xml_node, h5_id, path_prefix,
                               mesh->topology(), mesh->geometry(), dim,
                               active_entities);

  // Add attribute node with values
  pugi::xml_node attribute_node = xml_node.append_child("Attribute");
  assert(attribute_node);
  attribute_node.append_attribute("Name") = name.c_str();
  attribute_node.append_attribute("AttributeType") = "Scalar";
  attribute_node.append_attribute("Center") = "Cell";

  const std::int64_t global_num_values
      = dolfinx::MPI::sum(comm, (std::int64_t)meshtags.values().rows());

  const std::int64_t offset
      = dolfinx::MPI::global_offset(comm, meshtags.indices().rows(), true);
  const bool use_mpi_io = (dolfinx::MPI::size(comm) > 1);

  const std::vector<T> values(meshtags.values().data(),
                              meshtags.values().data()
                                  + meshtags.values().rows());
  xdmf_utils::add_data_item(attribute_node, h5_id, path_prefix + "/Values",
                            values, offset, {global_num_values, 1}, "",
                            use_mpi_io);
}

} // namespace xdmf_meshtags
} // namespace io
} // namespace dolfinx