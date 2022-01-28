// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <hdf5.h>
#include <mpi.h>
#include <string>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

namespace pugi
{
class xml_node;
} // namespace pugi

namespace dolfinx
{

namespace mesh
{
class Geometry;
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
void add_mesh(MPI_Comm comm, pugi::xml_node& xml_node, const hid_t h5_id,
              const mesh::Mesh& mesh, const std::string path_prefix);

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
void add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node,
                       const hid_t h5_id, const std::string path_prefix,
                       const mesh::Topology& topology,
                       const mesh::Geometry& geometry, int cell_dim,
                       const xtl::span<const std::int32_t>& entities);

/// Add Geometry xml node
void add_geometry_data(MPI_Comm comm, pugi::xml_node& xml_node,
                       const hid_t h5_id, const std::string path_prefix,
                       const mesh::Geometry& geometry);

/// Read Geometry data
/// @returns geometry
xt::xtensor<double, 2> read_geometry_data(MPI_Comm comm, const hid_t h5_id,
                                          const pugi::xml_node& node);

/// Read Topology data
/// @returns ((cell type, degree), topology)
xt::xtensor<std::int64_t, 2> read_topology_data(MPI_Comm comm,
                                                const hid_t h5_id,
                                                const pugi::xml_node& node);

} // namespace io::xdmf_mesh
} // namespace dolfinx
