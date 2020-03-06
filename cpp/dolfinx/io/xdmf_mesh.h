// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "HDF5File.h"
#include <Eigen/Dense>
#include <dolfinx/common/MPI.h>
#include <dolfinx/mesh/cell_types.h>
#include <string>
#include <tuple>
#include <vector>

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

namespace io
{
/// Low-level methods for reading XDMF files
namespace xdmf_mesh
{

/// TODO
void add_mesh(MPI_Comm comm, pugi::xml_node& xml_node, hid_t& h5_id,
              const mesh::Mesh& mesh, const std::string path_prefix);

/// TODO: Document
void add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node, hid_t& h5_id,
                       const std::string path_prefix,
                       const mesh::Topology& topology,
                       const mesh::Geometry& geometry, int cell_dim);

/// TODO
void add_geometry_data(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                       const std::string path_prefix,
                       const mesh::Geometry& geometry);

/// TODO
std::tuple<
    mesh::CellType,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
read_mesh_data(MPI_Comm comm, std::string filename);

} // namespace xdmf_mesh
} // namespace io
} // namespace dolfinx
