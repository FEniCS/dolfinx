// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_utils.h"
#include "pugixml.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <map>

using namespace dolfinx;
using namespace dolfinx::io;

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
         {"quadrilateral", {"quadrilateral", 1}},
         {"hexahedron", {"hexahedron", 1}}};

  // Convert XDMF cell type string to DOLFINX cell type string
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
                             + R"(", expecting "DataItem")");
  }

  // Check that format is HDF
  pugi::xml_attribute format_attr = dataitem_node.attribute("Format");
  assert(format_attr);
  const std::string format = format_attr.as_string();
  if (format.compare("HDF") != 0)
  {
    throw std::runtime_error("DataItem format \"" + format
                             + R"(" is not "HDF")");
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
    for (const auto& d : dims_list)
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
    throw std::runtime_error("Cannot determine number of cells in XMDF mesh");

  // Check for consistency if number of cells appears in both the topology
  // and DataItem nodes
  if (num_cells_topolgy != -1 and tdims.size() == 2)
  {
    if (num_cells_topolgy != tdims[0])
      throw std::runtime_error("Cannot determine number of cells in XMDF mesh");
  }

  return std::max(num_cells_topolgy, tdims[0]);
}
//----------------------------------------------------------------------------
std::vector<PetscScalar>
xdmf_utils::get_point_data_values(const function::Function& u)
{
  std::shared_ptr<const mesh::Mesh> mesh = u.function_space()->mesh();
  assert(mesh);
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      data_values = u.compute_point_values();

  const int width = get_padded_width(u);
  assert(mesh->geometry().index_map());
  const int num_local_points = mesh->geometry().index_map()->size_local();
  assert(data_values.rows() >= num_local_points);
  data_values.conservativeResize(num_local_points, Eigen::NoChange);

  // FIXME: Unpick the below code for the new layout of data from
  //        GenericFunction::compute_vertex_values
  std::vector<PetscScalar> _data_values(width * num_local_points, 0.0);
  const int value_rank = u.value_rank();
  if (value_rank > 0)
  {
    // Transpose vector/tensor data arrays
    const int value_size = u.value_size();
    for (int i = 0; i < num_local_points; i++)
    {
      for (int j = 0; j < value_size; j++)
      {
        int tensor_2d_offset
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

  return _data_values;
}
//-----------------------------------------------------------------------------
std::vector<PetscScalar>
xdmf_utils::get_cell_data_values(const function::Function& u)
{
  assert(u.function_space()->dofmap());
  const auto mesh = u.function_space()->mesh();
  const int value_size = u.value_size();
  const int value_rank = u.value_rank();

  // Allocate memory for function values at cell centres
  const int tdim = mesh->topology().dim();
  const std::int32_t num_local_cells
      = mesh->topology().index_map(tdim)->size_local();
  const std::int32_t local_size = num_local_cells * value_size;

  // Build lists of dofs and create map
  std::vector<std::int32_t> dof_set;
  dof_set.reserve(local_size);
  const auto dofmap = u.function_space()->dofmap();
  assert(dofmap->element_dof_layout);
  const int ndofs = dofmap->element_dof_layout->num_dofs();

  auto map_c = mesh->topology().index_map(0);
  assert(map_c);
  for (int cell = 0; map_c->size_local(); ++cell)
  {
    // Tabulate dofs
    auto dofs = dofmap->cell_dofs(cell);
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
                                          int num_nodes)
{
  static const std::map<mesh::CellType, std::map<int, std::string>> vtk_map = {
      {mesh::CellType::point, {{1, "PolyVertex"}}},
      {mesh::CellType::interval, {{2, "PolyLine"}, {3, "Edge_3"}}},
      {mesh::CellType::triangle, {{3, "Triangle"}, {6, "Triangle_6"}}},
      {mesh::CellType::quadrilateral,
       {{4, "Quadrilateral"}, {9, "Quadrilateral_9"}}},
      {mesh::CellType::tetrahedron,
       {{4, "Tetrahedron"}, {10, "Tetrahedron_10"}}},
      {mesh::CellType::hexahedron, {{8, "Hexahedron"}, {27, "Hexahedron_27"}}}};

  // Get cell family
  auto cell = vtk_map.find(cell_type);
  if (cell == vtk_map.end())
    throw std::runtime_error("Could not find cell type.");

  // Get cell string
  auto cell_str = cell->second.find(num_nodes);
  if (cell_str == cell->second.end())
    throw std::runtime_error("Could not find VTK string for cell order.");

  return cell_str->second;
}
//-----------------------------------------------------------------------------
