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
/// Convert a shape to the XDMF string description (Scalar, Vector,
/// Tensor).
std::string shape_to_string(std::span<const std::size_t> shape)
{
  if (shape.empty())
    return "Scalar";
  else if (shape.size() == 1 and shape[0] == 1)
    return "Scalar";
  else if (shape.size() == 1)
    return "Vector";
  else if (shape.size() == 2)
    return "Tensor";
  else
    throw std::runtime_error("Unsupported value shape");
}
} // namespace

//-----------------------------------------------------------------------------
template <dolfinx::scalar T, std::floating_point U>
void xdmf_function::add_function(MPI_Comm comm, const fem::Function<T, U>& u,
                                 double t, pugi::xml_node& xml_node,
                                 hid_t h5_id)
{
  spdlog::info("Adding function to node \"{}\"", xml_node.path('/'));

  assert(u.function_space());
  auto mesh = u.function_space()->mesh();
  assert(mesh);
  std::shared_ptr<const fem::FiniteElement<U>> element
      = u.function_space()->element();
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

  auto map_x = mesh->geometry().index_map();
  assert(map_x);

  auto dofmap = u.function_space()->dofmap();
  assert(dofmap);
  const int bs = dofmap->bs();

  // Pad to 3D if vector/tensor is product of dimensions is smaller than 3**rank
  // to ensure that we can visualize them correctly in Paraview
  std::span<const std::size_t> value_shape = u.function_space()->value_shape();
  int rank = value_shape.size();
  int num_components = std::reduce(value_shape.begin(), value_shape.end(), 1,
                                   std::multiplies{});
  if (num_components < std::pow(3, rank))
    num_components = std::pow(3, rank);

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
    auto& cmap = geometry.cmap();
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
    std::int32_t num_local_points = map_x->size_local();

    // Get dof array and pack into array (padded where appropriate)
    auto dofmap_x = geometry.dofmap();
    data_values.resize(num_local_points * num_components, 0);
    for (std::int32_t c = 0; c < num_cells; ++c)
    {
      auto dofs = dofmap->cell_dofs(c);
      auto dofs_x = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          dofmap_x, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
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

  // Global size
  const std::int64_t num_values
      = cell_centred ? map_c->size_global() : map_x->size_global();

  const std::int64_t num_local = data_values.size() / num_components;
  std::int64_t offset = 0;
  MPI_Exscan(&num_local, &offset, 1, MPI_INT64_T, MPI_SUM, comm);

  const bool use_mpi_io = dolfinx::MPI::size(comm) > 1;

  std::vector<std::string> components = {""};
  if constexpr (!std::is_scalar_v<T>)
    components = {"real_", "imag_"};
  std::string t_str = boost::lexical_cast<std::string>(t);
  std::ranges::replace(t_str, '.', '_');
  for (auto component : components)
  {
    std::string attr_name = component + u.name;
    std::string dataset_name
        = std::string("/Function/") + attr_name + std::string("/") + t_str;

    // Add attribute node
    pugi::xml_node attr_node = xml_node.append_child("Attribute");
    assert(attr_node);
    attr_node.append_attribute("Name") = attr_name.c_str();
    attr_node.append_attribute("AttributeType")
        = shape_to_string(value_shape).c_str();
    attr_node.append_attribute("Center") = cell_centred ? "Cell" : "Node";

    std::span<const scalar_value_type_t<T>> u;
    std::vector<scalar_value_type_t<T>> _data;
    if constexpr (!std::is_scalar_v<T>)
    {
      // Complex-valued case
      _data.resize(data_values.size());
      if (component == "real_")
      {
        std::ranges::transform(data_values, _data.begin(),
                               [](auto x) { return x.real(); });
      }
      else if (component == "imag_")
      {
        std::ranges::transform(data_values, _data.begin(),
                               [](auto x) { return x.imag(); });
      }
      u = std::span<const scalar_value_type_t<T>>(_data);
    }
    else
      u = std::span<const T>(data_values);

    // -- Real case, add data item
    xdmf_utils::add_data_item(attr_node, h5_id, dataset_name, u, offset,
                              {num_values, num_components}, "", use_mpi_io);
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
