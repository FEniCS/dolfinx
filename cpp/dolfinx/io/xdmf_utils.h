// Copyright (C) 2012 Chris N. Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

// #include "HDF5Interface.h"
// #include <dolfinx/common/MPI.h>
// #include <dolfinx/function/Function.h>
// #include <dolfinx/la/PETScVector.h>
// #include <dolfinx/mesh/Mesh.h>
// #include <dolfinx/mesh/MeshFunction.h>
// #include <dolfinx/mesh/MeshValueCollection.h>
// #include <memory>
#include <array>
#include <dolfinx/mesh/cell_types.h>
#include <petscsys.h>
#include <string>
#include <utility>
#include <vector>

namespace pugi
{
class xml_node;
} // namespace pugi

namespace dolfinx
{

namespace function
{
class Function;
} // namespace function

// namespace mesh
// {
// class CellType;
// } // namespace mesh

namespace io
{
namespace xdmf_utils
{

// Get DOLFINX cell type string from XML topology node
std::pair<std::string, int> get_cell_type(const pugi::xml_node& topology_node);

// Return (0) HDF5 filename and (1) path in HDF5 file from a DataItem
// node
std::array<std::string, 2> get_hdf5_paths(const pugi::xml_node& dataitem_node);

std::string get_hdf5_filename(std::string xdmf_filename);

/// Get dimensions from an XML DataSet node
std::vector<std::int64_t> get_dataset_shape(const pugi::xml_node& dataset_node);

/// Get number of cells from an XML Topology node
std::int64_t get_num_cells(const pugi::xml_node& topology_node);

/// Get point data values for linear or quadratic mesh into flattened 2D
/// array
std::vector<PetscScalar> get_point_data_values(const function::Function& u);

/// Get cell data values as a flattened 2D array
std::vector<PetscScalar> get_cell_data_values(const function::Function& u);

std::string vtk_cell_type_str(mesh::CellType cell_type, int order);

} // namespace xdmf_utils
} // namespace io
} // namespace dolfinx
