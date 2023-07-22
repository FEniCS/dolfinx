// Copyright (C) 2012-2020 Chris N. Richardson, Garth N. Wells and Michal Habera
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_function.h"
#include "xdmf_mesh.h"
#include "xdmf_utils.h"
#include <basix/mdspan.hpp>
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

/// Convert a value_rank to the XDMF string description (Scalar, Vector,
/// Tensor).
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

/// Get data width - normally the same as u.value_size(), but expand for
/// 2D vector/tensor because XDMF presents everything as 3D
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

  // FIXME: is the below check adequate for detecting a Lagrange
  // element?
  // Check that element is Lagrange
  if (!element->interpolation_ident())
  {
    throw std::runtime_error("Only Lagrange functions are supported. "
                             "Interpolate Functions before output.");
  }

  auto map_c = mesh->topology()->index_map(mesh->topology()->dim());
  assert(map_c);

  auto dofmap = u.function_space()->dofmap();
  assert(dofmap);
  const int bs = dofmap->bs();

  int rank = element->value_shape().size();
  int num_components = std::pow(3, rank);

  // Get fem::Function data values and shape
  std::vector<T> data_values;
  std::span<const T> x = u.x()->array();

  const bool cell_centred
      = element->space_dimension() / element->block_size() == 1;
  if (cell_centred)
  {
    // Get dof array and pack into array (padded where appropriate)
    const std::int32_t num_local_cells = map_c->size_local();
    data_values.resize(num_local_cells * num_components, 0);
    for (std::int32_t c = 0; c < num_local_cells; ++c)
    {
      auto dofs = dofmap->cell_dofs(c);
      assert(dofs.size() == 1);
      for (std::size_t i = 0; i < dofs.size(); ++i)
        for (int j = 0; j < bs; ++j)
          data_values[num_components * c + j] = x[bs * dofs[i] + j];
    }
  }
  else
  {
    // Get number of geometry nodes per cell
    const auto& geometry = mesh->geometry();
    auto& cmap = geometry.cmaps()[0];
    int cmap_dim = cmap.dim();
    int cell_dim = element->space_dimension() / element->block_size();
    if (cmap_dim != cell_dim)
    {
      throw std::runtime_error(
          "Degree of output Function must be same as mesh degree. Maybe the "
          "Function needs to be interpolated?");
    }

    // Check that dofmap layouts are equal and check Lagrange variants
    if (dofmap->element_dof_layout() != cmap.create_dof_layout())
    {
      throw std::runtime_error("Function and Mesh dof layouts do not match. "
                               "Maybe the Function needs to be interpolated?");
    }
    if (cmap.degree() > 2
        and element->basix_element().lagrange_variant() != cmap.variant())
    {
      throw std::runtime_error("Mis-match in Lagrange family. Maybe the "
                               "Function needs to be interpolated?");
    }

    std::int32_t num_cells = map_c->size_local() + map_c->num_ghosts();
    std::int32_t num_local_points = geometry.index_map()->size_local();

    // Get dof array and pack into array (padded where appropriate)
    namespace stdex = std::experimental;
    auto dofmap_x = geometry.dofmap();
    data_values.resize(num_local_points * num_components, 0);
    for (std::int32_t c = 0; c < num_cells; ++c)
    {
      auto dofs = dofmap->cell_dofs(c);
      auto dofs_x = stdex::submdspan(dofmap_x, c, stdex::full_extent);
      assert(dofs.size() == dofs_x.size());
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        if (dofs_x[i] < num_local_points)
        {
          for (int j = 0; j < bs; ++j)
            data_values[num_components * dofs_x[i] + j] = x[bs * dofs[i] + j];
        }
      }
    }
  }

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
