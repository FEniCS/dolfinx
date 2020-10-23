// Copyright (C) 2012-2020 Chris N. Richardson, Garth N. Wells and Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_function.h"
#include "pugixml.hpp"
#include "xdmf_mesh.h"
#include "xdmf_utils.h"
#include <boost/lexical_cast.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <string>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
//-----------------------------------------------------------------------------

// Convert a value_rank to the XDMF string description (Scalar, Vector,
// Tensor)
std::string rank_to_string(int value_rank)
{
  switch (value_rank)
  {
  case 0:
    return "Scalar";
  case 1:
    return "Vector";
  case 2:
    return "Tensor";
  default:
    throw std::runtime_error("Range Error");
  }
}
//-----------------------------------------------------------------------------

/// Returns true for DG0 function::Functions
bool has_cell_centred_data(const function::Function<PetscScalar>& u)
{
  int cell_based_dim = 1;
  const int rank = u.function_space()->element()->value_rank();
  for (int i = 0; i < rank; i++)
    cell_based_dim *= u.function_space()->mesh()->topology().dim();

  assert(u.function_space());
  assert(u.function_space()->dofmap());
  assert(u.function_space()->dofmap()->element_dof_layout);
  return (u.function_space()->dofmap()->element_dof_layout->num_dofs()
          == cell_based_dim);
}
//-----------------------------------------------------------------------------

// Get data width - normally the same as u.value_size(), but expand for
// 2D vector/tensor because XDMF presents everything as 3D
int get_padded_width(const function::Function<PetscScalar>& u)
{
  const int width = u.function_space()->element()->value_size();
  const int rank = u.function_space()->element()->value_rank();
  if (rank == 1 and width == 2)
    return 3;
  else if (rank == 2 and width == 4)
    return 9;
  return width;
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
void xdmf_function::add_function(MPI_Comm comm,
                                 const function::Function<PetscScalar>& u,
                                 const double t, pugi::xml_node& xml_node,
                                 const hid_t h5_id)
{
  LOG(INFO) << "Adding function to node \"" << xml_node.path('/') << "\"";

  assert(u.function_space());
  std::shared_ptr<const mesh::Mesh> mesh = u.function_space()->mesh();
  assert(mesh);

  // Get function::Function data values and shape
  std::vector<PetscScalar> data_values;
  const bool cell_centred = has_cell_centred_data(u);
  if (cell_centred)
    data_values = xdmf_utils::get_cell_data_values(u);
  else
    data_values = xdmf_utils::get_point_data_values(u);

  auto map_c = mesh->topology().index_map(mesh->topology().dim());
  assert(map_c);

  auto map_v = mesh->geometry().index_map();
  assert(map_v);

  // Add attribute DataItem node and write data
  const int width = get_padded_width(u);
  assert(data_values.size() % width == 0);
  const int num_values
      = cell_centred ? map_c->size_global() : map_v->size_global();

  const int value_rank = u.function_space()->element()->value_rank();

#ifdef PETSC_USE_COMPLEX
  const std::vector<std::string> components = {"real", "imag"};
#else
  const std::vector<std::string> components = {""};
#endif

  std::string t_str = boost::lexical_cast<std::string>(t);
  std::replace(t_str.begin(), t_str.end(), '.', '_');

  for (const auto& component : components)
  {
    std::string attr_name;
    std::string dataset_name;
    if (component.empty())
    {
      attr_name = u.name;
      dataset_name = "/Function/" + attr_name + "/" + t_str;
    }
    else
    {
      attr_name = component + "_" + u.name;
      dataset_name = "/Function/" + attr_name + "/" + t_str;
    }
    // Add attribute node
    pugi::xml_node attribute_node = xml_node.append_child("Attribute");
    assert(attribute_node);
    attribute_node.append_attribute("Name") = attr_name.c_str();
    attribute_node.append_attribute("AttributeType")
        = rank_to_string(value_rank).c_str();
    attribute_node.append_attribute("Center") = cell_centred ? "Cell" : "Node";

    const bool use_mpi_io = (dolfinx::MPI::size(comm) > 1);
#ifdef PETSC_USE_COMPLEX
    // FIXME: Avoid copies by writing directly a compound data
    std::vector<double> component_data_values(data_values.size());
    if (component == "real")
    {
      for (std::size_t i = 0; i < data_values.size(); i++)
        component_data_values[i] = data_values[i].real();
    }
    else if (component == "imag")
    {
      for (std::size_t i = 0; i < data_values.size(); i++)
        component_data_values[i] = data_values[i].imag();
    }

    // Add data item of component
    const std::int64_t offset = dolfinx::MPI::global_offset(
        comm, component_data_values.size() / width, true);
    xdmf_utils::add_data_item(attribute_node, h5_id, dataset_name,
                              component_data_values, offset,
                              {num_values, width}, "", use_mpi_io);
#else
    // Add data item
    const std::int64_t offset
        = dolfinx::MPI::global_offset(comm, data_values.size() / width, true);
    xdmf_utils::add_data_item(attribute_node, h5_id, dataset_name, data_values,
                              offset, {num_values, width}, "", use_mpi_io);
#endif
  }
}
//-----------------------------------------------------------------------------
