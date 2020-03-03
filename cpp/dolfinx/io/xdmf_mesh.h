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
