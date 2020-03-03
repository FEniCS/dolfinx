// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/common/MPI.h>
#include <dolfinx/mesh/cell_types.h>
#include <string>
#include <tuple>
#include <vector>

// #include <dolfinx/mesh/MeshIterator.h>

// #include "HDF5File.h"
// #include "pugixml.hpp"
// #include "xdmf_utils.h"
// #include <boost/algorithm/string.hpp>
// #include <boost/filesystem.hpp>
// #include <boost/lexical_cast.hpp>
// #include <dolfinx/mesh/MeshEntity.h>
// #include <dolfinx/mesh/MeshFunction.h>
// #include <dolfinx/mesh/MeshIterator.h>

namespace dolfinx
{

namespace io
{
/// Low-level methods for reading XDMF files
namespace xdmf_mesh
{

/// TODO
std::tuple<
    mesh::CellType,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
read_mesh_data(MPI_Comm comm, std::string filename);

} // namespace xdmf_mesh
} // namespace io
} // namespace dolfinx
