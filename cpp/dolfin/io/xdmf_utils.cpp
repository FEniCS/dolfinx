// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_utils.h"

// #include "HDF5File.h"
// #include "HDF5Utility.h"
// #include "XDMFFile.h"
#include "pugixml.hpp"
// #include <algorithm>
#include <boost/algorithm/string.hpp>
// #include <boost/container/vector.hpp>
#include <boost/filesystem.hpp>
// #include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
// #include <dolfin/common/MPI.h>
// #include <dolfin/common/defines.h>
// #include <dolfin/common/log.h>
// #include <dolfin/common/utils.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
// #include <dolfin/la/PETScVector.h>
// #include <dolfin/la/utils.h>
#include <dolfin/mesh/Cell.h>
// #include <dolfin/mesh/Connectivity.h>
#include <dolfin/mesh/DistributedMeshTools.h>
// #include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/cell_types.h>
// #include <dolfin/mesh/MeshValueCollection.h>
// #include <dolfin/mesh/Partitioning.h>
// #include <iomanip>
// #include <memory>
// #include <petscvec.h>
// #include <set>
// #include <string>
// #include <vector>
#include <map>

using namespace dolfin;
using namespace dolfin::io;

namespace
{
// Get data width - normally the same as u.value_size(), but expand for
// 2D vector/tensor because XDMF presents everything as 3D
std::int64_t get_padded_width(const function::Function& u)
{
  std::int64_t width = u.value_size();
  std::int64_t rank = u.value_rank();
  if (rank == 1 and width == 2)
    return 3;
  else if (rank == 2 and width == 4)
    return 9;
  return width;
}
//-----------------------------------------------------------------------------

} // namespace

//----------------------------------------------------------------------------
std::pair<std::string, int>
xdmf_utils::get_cell_type(const pugi::xml_node& topology_node)
{
  assert(topology_node);
  pugi::xml_attribute type_attr = topology_node.attribute("TopologyType");
  assert(type_attr);

  const std::map<std::string, std::pair<std::string, int>> xdmf_to_dolfin
      = {{"polyvertex", {"point", 1}},
         {"polyline", {"interval", 1}},
         {"edge_3", {"interval", 2}},
         {"triangle", {"triangle", 1}},
         {"triangle_6", {"triangle", 2}},
         {"tetrahedron", {"tetrahedron", 1}},
         {"tetrahedron_10", {"tetrahedron", 2}},
         {"quadrilateral", {"quadrilateral", 1}}};

  // Convert XDMF cell type string to DOLFIN cell type string
  std::string cell_type = type_attr.as_string();
  boost::algorithm::to_lower(cell_type);
  auto it = xdmf_to_dolfin.find(cell_type);
  if (it == xdmf_to_dolfin.end())
  {
    throw std::runtime_error("Cannot recognise cell type. Unknown value: "
                             + cell_type);
  }
  return it->second;
}
//----------------------------------------------------------------------------
std::array<std::string, 2>
xdmf_utils::get_hdf5_paths(const pugi::xml_node& dataitem_node)
{
  // Check that node is a DataItem node
  assert(dataitem_node);
  const std::string dataitem_str = "DataItem";
  if (dataitem_node.name() != dataitem_str)
  {
    throw std::runtime_error("Node name is \""
                             + std::string(dataitem_node.name())
                             + "\", expecting \"DataItem\"");
  }

  // Check that format is HDF
  pugi::xml_attribute format_attr = dataitem_node.attribute("Format");
  assert(format_attr);
  const std::string format = format_attr.as_string();
  if (format.compare("HDF") != 0)
  {
    throw std::runtime_error("DataItem format \"" + format
                             + "\" is not \"HDF\"");
  }

  // Get path data
  pugi::xml_node path_node = dataitem_node.first_child();
  assert(path_node);

  // Create string from path and trim leading and trailing whitespace
  std::string path = path_node.text().get();
  boost::algorithm::trim(path);

  // Split string into file path and HD5 internal path
  std::vector<std::string> paths;
  boost::split(paths, path, boost::is_any_of(":"));
  assert(paths.size() == 2);

  return {{paths[0], paths[1]}};
}
//-----------------------------------------------------------------------------
std::string xdmf_utils::get_hdf5_filename(std::string xdmf_filename)
{
  boost::filesystem::path p(xdmf_filename);
  p.replace_extension(".h5");
  if (p.string() == xdmf_filename)
  {
    throw std::runtime_error("Cannot deduce name of HDF5 file from XDMF "
                             "filename. Filename clash. Check XDMF filename");
  }

  return p.string();
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
xdmf_utils::get_dataset_shape(const pugi::xml_node& dataset_node)
{
  // Get Dimensions attribute string
  assert(dataset_node);
  pugi::xml_attribute dimensions_attr = dataset_node.attribute("Dimensions");

  // Gets dimensions, if attribute is present
  std::vector<std::int64_t> dims;
  if (dimensions_attr)
  {
    // Split dimensions string
    const std::string dims_str = dimensions_attr.as_string();
    std::vector<std::string> dims_list;
    boost::split(dims_list, dims_str, boost::is_any_of(" "));

    // Cast dims to integers
    for (auto d : dims_list)
      dims.push_back(boost::lexical_cast<std::int64_t>(d));
  }

  return dims;
}
//----------------------------------------------------------------------------
std::int64_t xdmf_utils::get_num_cells(const pugi::xml_node& topology_node)
{
  assert(topology_node);

  // Get number of cells from topology
  std::int64_t num_cells_topolgy = -1;
  pugi::xml_attribute num_cells_attr
      = topology_node.attribute("NumberOfElements");
  if (num_cells_attr)
    num_cells_topolgy = num_cells_attr.as_llong();

  // Get number of cells from topology dataset
  pugi::xml_node topology_dataset_node = topology_node.child("DataItem");
  assert(topology_dataset_node);
  const std::vector<std::int64_t> tdims
      = get_dataset_shape(topology_dataset_node);

  // Check that number of cells can be determined
  if (tdims.size() != 2 and num_cells_topolgy == -1)
  {
    throw std::runtime_error("Cannot determine number of cells in XMDF mesh");
  }

  // Check for consistency if number of cells appears in both the topology
  // and DataItem nodes
  if (num_cells_topolgy != -1 and tdims.size() == 2)
  {
    if (num_cells_topolgy != tdims[0])
    {
      throw std::runtime_error("Cannot determine number of cells in XMDF mesh");
    }
  }

  return std::max(num_cells_topolgy, tdims[0]);
}
//----------------------------------------------------------------------------
std::vector<PetscScalar>
xdmf_utils::get_point_data_values(const function::Function& u)
{
  auto mesh = u.function_space()->mesh;
  assert(mesh);
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      data_values = u.compute_point_values(*mesh);
  std::int64_t width = get_padded_width(u);

  // FIXME: Unpick the below code for the new layout of data from
  //        GenericFunction::compute_vertex_values
  const std::size_t num_local_points = mesh->geometry().num_points();
  std::vector<PetscScalar> _data_values(width * num_local_points, 0.0);

  const std::size_t value_rank = u.value_rank();
  if (value_rank > 0)
  {
    // Transpose vector/tensor data arrays
    const std::size_t value_size = u.value_size();
    for (std::size_t i = 0; i < num_local_points; i++)
    {
      for (std::size_t j = 0; j < value_size; j++)
      {
        std::size_t tensor_2d_offset
            = (j > 1 && value_rank == 2 && value_size == 4) ? 1 : 0;
        _data_values[i * width + j + tensor_2d_offset] = data_values(i, j);
      }
    }
  }
  else
  {
    _data_values = std::vector<PetscScalar>(
        data_values.data(),
        data_values.data() + data_values.rows() * data_values.cols());
  }

  // Reorder values by global point indices
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>>
      in_vals(_data_values.data(), _data_values.size() / width, width);

  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      vals = mesh::DistributedMeshTools::reorder_by_global_indices(
          mesh->mpi_comm(), in_vals, mesh->geometry().global_indices());

  _data_values
      = std::vector<PetscScalar>(vals.data(), vals.data() + vals.size());

  return _data_values;
}
//-----------------------------------------------------------------------------
std::vector<PetscScalar>
xdmf_utils::get_cell_data_values(const function::Function& u)
{
  assert(u.function_space()->dofmap);
  const auto mesh = u.function_space()->mesh;
  const int value_size = u.value_size();
  const int value_rank = u.value_rank();

  // Allocate memory for function values at cell centres
  const int tdim = mesh->topology().dim();
  const std::int32_t num_local_cells = mesh->topology().ghost_offset(tdim);
  const std::int32_t local_size = num_local_cells * value_size;

  // Build lists of dofs and create map
  std::vector<PetscInt> dof_set;
  dof_set.reserve(local_size);
  const auto dofmap = u.function_space()->dofmap;
  assert(dofmap->element_dof_layout);
  const int ndofs = dofmap->element_dof_layout->num_dofs();
  for (auto& cell : mesh::MeshRange<mesh::Cell>(*mesh))
  {
    // Tabulate dofs
    auto dofs = dofmap->cell_dofs(cell.index());
    assert(ndofs == value_size);
    for (int i = 0; i < ndofs; ++i)
      dof_set.push_back(dofs[i]);
  }

  // Get  values
  std::vector<PetscScalar> data_values(dof_set.size());
  {
    la::VecReadWrapper u_wrapper(u.vector().vec());
    Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x
        = u_wrapper.x;
    for (std::size_t i = 0; i < dof_set.size(); ++i)
      data_values[i] = x[dof_set[i]];
  }

  if (value_rank == 1 && value_size == 2)
  {
    // Pad out data for 2D vector to 3D
    data_values.resize(3 * num_local_cells);
    for (int j = (num_local_cells - 1); j >= 0; --j)
    {
      PetscScalar nd[3] = {data_values[j * 2], data_values[j * 2 + 1], 0};
      std::copy(nd, nd + 3, &data_values[j * 3]);
    }
  }
  else if (value_rank == 2 && value_size == 4)
  {
    data_values.resize(9 * num_local_cells);
    for (int j = (num_local_cells - 1); j >= 0; --j)
    {
      PetscScalar nd[9] = {data_values[j * 4],
                           data_values[j * 4 + 1],
                           0,
                           data_values[j * 4 + 2],
                           data_values[j * 4 + 3],
                           0,
                           0,
                           0,
                           0};
      std::copy(nd, nd + 9, &data_values[j * 9]);
    }
  }
  return data_values;
}
//-----------------------------------------------------------------------------
std::string xdmf_utils::vtk_cell_type_str(mesh::CellType cell_type,
                                          int order)
{
  // FIXME: Move to CellType?
  switch (cell_type)
  {
  case mesh::CellType::point:
    switch (order)
    {
    case 1:
      return "PolyVertex";
    }
  case mesh::CellType::interval:
    switch (order)
    {
    case 1:
      return "PolyLine";
    case 2:
      return "Edge_3";
    }
  case mesh::CellType::triangle:
    switch (order)
    {
    case 1:
      return "Triangle";
    case 2:
      return "Triangle_6";
    }
  case mesh::CellType::quadrilateral:
    switch (order)
    {
    case 1:
      return "Quadrilateral";
    case 2:
      return "Quadrilateral_8";
    }
  case mesh::CellType::tetrahedron:
    switch (order)
    {
    case 1:
      return "Tetrahedron";
    case 2:
      return "Tetrahedron_10";
    }
  case mesh::CellType::hexahedron:
    switch (order)
    {
    case 1:
      return "Hexahedron";
    case 2:
      return "Hexahedron_20";
    }
  default:
    throw std::runtime_error("Invalid combination of cell type and order");
    return "error";
  }
}
//-----------------------------------------------------------------------------
