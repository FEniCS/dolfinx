// Copyright (C) 2012 Chris N. Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

// #include "HDF5Interface.h"
// #include <dolfin/common/MPI.h>
// #include <dolfin/function/Function.h>
// #include <dolfin/la/PETScVector.h>
// #include <dolfin/mesh/Mesh.h>
// #include <dolfin/mesh/MeshFunction.h>
// #include <dolfin/mesh/MeshValueCollection.h>
// #include <memory>
#include <string>
// #include <utility>
#include <dolfin/mesh/CellType.h>
#include <petscsys.h>
#include <vector>

namespace pugi
{
class xml_node;
} // namespace pugi

namespace dolfin
{

namespace function
{
class Function;
} // namespace function

// namespace mesh
// {
// class CellType::Type;
// } // namespace mesh

namespace io
{
namespace xdmf_utils
{

/// Get dimensions from an XML DataSet node
std::vector<std::int64_t> get_dataset_shape(const pugi::xml_node& dataset_node);

/// Get number of cells from an XML Topology node
std::int64_t get_num_cells(const pugi::xml_node& topology_node);

/// Get point data values for linear or quadratic mesh into flattened 2D
/// array
std::vector<PetscScalar> get_point_data_values(const function::Function& u);

/// Get cell data values as a flattened 2D array
std::vector<PetscScalar> get_cell_data_values(const function::Function& u);

std::string vtk_cell_type_str(mesh::CellType::Type cell_type, int order);

} // namespace xdmf_utils
} // namespace io
} // namespace dolfin
