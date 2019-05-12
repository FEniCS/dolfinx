// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

// #include <cstdint>
#include <dolfin/common/MPI.h>
// #include <dolfin/mesh/CellType.h>
#include <hdf5.h>
// #include <memory>
// #include <petscsys.h>
// #include <string>
// #include <utility>
// #include <vector>

namespace pugi
{
class xml_node;
} // namespace pugi

namespace dolfin
{
namespace geometry
{
class Point;
}
namespace function
{
class Function;
}
namespace mesh
{
class Mesh;
}

namespace io
{
namespace xdmf_write
{

/// Add set of points to XDMF xml_node and write data
void add_points(MPI_Comm comm, pugi::xml_node& xdmf_node, hid_t h5_id,
                const std::vector<geometry::Point>& points);

/// Add topology node to xml_node (includes writing data to XML or HDF5
/// file)
void add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                       const std::string path_prefix, const mesh::Mesh& mesh,
                       int cell_dim);

/// Add geometry node and data to xml_node
void add_geometry_data(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                       const std::string path_prefix, const mesh::Mesh& mesh);

/// Add mesh to XDMF xml_node (usually a Domain or Time Grid) and write
/// data
void add_mesh(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
              const mesh::Mesh& mesh, const std::string path_prefix);

// Add function to a XML node
void add_function(MPI_Comm mpi_comm, pugi::xml_node& xml_node, hid_t h5_id,
                  std::string h5_path, const function::Function& u,
                  std::string function_name, const mesh::Mesh& mesh,
                  const std::string component);

} // namespace xdmf_write
} // namespace io
} // namespace dolfin
