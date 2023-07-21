// Copyright (C) 2012-2020 Chris N. Richardson, Garth N. Wells and Michal Habera
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_function.h"
#include "xdmf_mesh.h"
#include "xdmf_utils.h"
#include <boost/lexical_cast.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <pugixml.hpp>
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

/// Returns true for DG0 fem::Functions
template <typename Scalar>
bool has_cell_centred_data(
    const fem::Function<Scalar, dolfinx::scalar_value_type_t<Scalar>>& u)
{
  int cell_based_dim = 1;
  const int rank = u.function_space()->element()->value_shape().size();
  for (int i = 0; i < rank; i++)
    cell_based_dim *= u.function_space()->mesh()->topology()->dim();

  assert(u.function_space());
  assert(u.function_space()->dofmap());
  return (u.function_space()->dofmap()->element_dof_layout().num_dofs()
              * u.function_space()->dofmap()->bs()
          == cell_based_dim);
}
//-----------------------------------------------------------------------------

// Get data width - normally the same as u.value_size(), but expand for
// 2D vector/tensor because XDMF presents everything as 3D
template <std::floating_point U>
int get_padded_width(const fem::FiniteElement<U>& e)
{
  const int width = e.value_size();
  const int rank = e.value_shape().size();
  if (rank == 1 and width == 2)
    return 3;
  else if (rank == 2 and width == 4)
    return 9;
  return width;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
template <dolfinx::scalar T, std::floating_point U>
void xdmf_function::add_function(MPI_Comm comm, const fem::Function<T, U>& u,
                                 double t, pugi::xml_node& xml_node,
                                 hid_t h5_id)
{
  LOG(INFO) << "Adding function to node \"" << xml_node.path('/') << "\"";

  assert(u.function_space());
  auto mesh = u.function_space()->mesh();
  assert(mesh);
  auto element = u.function_space()->element();
  assert(element);

  // FIXME: is the below check adequate for detecting a
  // Lagrange element? Check that element is Lagrange
  if (!element->interpolation_ident())
  {
    throw std::runtime_error("Only Lagrange functions are supported. "
                             "Interpolate Functions before output.");
  }

  // Get fem::Function data values and shape
  std::vector<T> data_values;
  const bool cell_centred = has_cell_centred_data(u);
  if (cell_centred)
    data_values = xdmf_utils::get_cell_data_values(u);
  else
  {
    auto cell_types = mesh->topology()->cell_types();
    int num_vertices_per_cell = mesh::cell_num_entities(cell_types.back(), 0);
    assert(u.function_space()->dofmap());
    auto dof_layout = u.function_space()->dofmap()->element_dof_layout();
    int num_vertex_dofs = dof_layout.num_entity_dofs(0);
    int num_dofs = element->space_dimension() / element->block_size();
    if (num_dofs != num_vertices_per_cell or num_vertex_dofs != 1)
    {
      throw std::runtime_error(
          "Only first order Lagrange spaces are supported with XDMF.");
    }

    // bool need_padding = rank > 0 and gdim != 3 ? true : false;
    // auto eq_check = [](auto& x, auto& y) -> bool
    // {
    //   return x.extents() == y.extents()
    //          and std::equal(x.data_handle(), x.data_handle() + x.size(),
    //                         y.data_handle());
    // };
    // if (!need_padding
    //     and eq_check(mesh->geometry().dofmap(),
    //                  u.function_space()->dofmap->map()))
    // {
    //   data_values.assign(u.x()->array().begin(), u.x()->array().end());
    // }
    // else
    // {

    // }

    data_values = xdmf_utils::get_point_data_values(u);
  }

  auto map_c = mesh->topology()->index_map(mesh->topology()->dim());
  assert(map_c);

  auto map_v = mesh->geometry().index_map();
  assert(map_v);

  // Add attribute DataItem node and write data
  const int width = get_padded_width(*u.function_space()->element());
  assert(data_values.size() % width == 0);
  const std::int64_t num_values
      = cell_centred ? map_c->size_global() : map_v->size_global();

  const int value_rank = u.function_space()->element()->value_shape().size();

  std::vector<std::string> components = {""};
  if constexpr (!std::is_scalar_v<T>)
    components = {"real", "imag"};

  std::string t_str = boost::lexical_cast<std::string>(t);
  std::replace(t_str.begin(), t_str.end(), '.', '_');

  for (const auto& component : components)
  {
    std::string attr_name;
    std::string dataset_name;
    if (component.empty())
    {
      attr_name = u.name;
      dataset_name
          = std::string("/Function/") + attr_name + std::string("/") + t_str;
    }
    else
    {
      attr_name = component + std::string("_") + u.name;
      dataset_name
          = std::string("/Function/") + attr_name + std::string("/") + t_str;
    }
    // Add attribute node
    pugi::xml_node attribute_node = xml_node.append_child("Attribute");
    assert(attribute_node);
    attribute_node.append_attribute("Name") = attr_name.c_str();
    attribute_node.append_attribute("AttributeType")
        = rank_to_string(value_rank).c_str();
    attribute_node.append_attribute("Center") = cell_centred ? "Cell" : "Node";

    const bool use_mpi_io = dolfinx::MPI::size(comm) > 1;
    if constexpr (!std::is_scalar_v<T>)
    {
      // Complex case

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
      const std::int64_t num_local = component_data_values.size() / width;
      std::int64_t offset = 0;
      MPI_Exscan(&num_local, &offset, 1, MPI_INT64_T, MPI_SUM, comm);
      xdmf_utils::add_data_item(attribute_node, h5_id, dataset_name,
                                component_data_values, offset,
                                {num_values, width}, "", use_mpi_io);
    }
    else
    {
      // Real case

      // Add data item
      const std::int64_t num_local = data_values.size() / width;
      std::int64_t offset = 0;
      MPI_Exscan(&num_local, &offset, 1, MPI_INT64_T, MPI_SUM, comm);
      xdmf_utils::add_data_item(attribute_node, h5_id, dataset_name,
                                data_values, offset, {num_values, width}, "",
                                use_mpi_io);
    }
  }
}
//-----------------------------------------------------------------------------
// Instantiation for different types
/// @cond
template void xdmf_function::add_function(MPI_Comm,
                                          const fem::Function<float, float>&,
                                          double, pugi::xml_node&, hid_t);
template void xdmf_function::add_function(MPI_Comm,
                                          const fem::Function<double, double>&,
                                          double, pugi::xml_node&, hid_t);
template void
xdmf_function::add_function(MPI_Comm,
                            const fem::Function<std::complex<float>, float>&,
                            double, pugi::xml_node&, hid_t);
template void
xdmf_function::add_function(MPI_Comm,
                            const fem::Function<std::complex<double>, double>&,
                            double, pugi::xml_node&, hid_t);

/// @endcond
//-----------------------------------------------------------------------------
