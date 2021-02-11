// Copyright (C) 2013-2020 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CoordinateElement.h"
#include "DofMap.h"
#include "ElementDofLayout.h"
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/cell_types.h>
#include <memory>
#include <set>
#include <string>
#include <ufc.h>
#include <utility>
#include <vector>

namespace dolfinx
{
namespace common
{
class IndexMap;
}

namespace mesh
{
class Mesh;
class Topology;
} // namespace mesh

namespace fem
{

template <typename T>
class Constant;
template <typename T>
class Form;
template <typename T>
class Function;
class FunctionSpace;

/// Extract test (0) and trial (1) function spaces pairs for each
/// bilinear form for a rectangular array of forms
///
/// @param[in] a A rectangular block on bilinear forms
/// @return Rectangular array of the same shape as @p a with a pair of
///   function spaces in each array entry. If a form is null, then the
///   returned function space pair is (null, null).
template <typename T>
std::vector<
    std::vector<std::array<std::shared_ptr<const fem::FunctionSpace>, 2>>>
extract_function_spaces(const std::vector<std::vector<const fem::Form<T>*>>& a)
{
  std::vector<
      std::vector<std::array<std::shared_ptr<const fem::FunctionSpace>, 2>>>
      spaces(
          a.size(),
          std::vector<std::array<std::shared_ptr<const fem::FunctionSpace>, 2>>(
              a[0].size()));
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < a[i].size(); ++j)
    {
      if (const fem::Form<T>* form = a[i][j]; form)
        spaces[i][j] = {form->function_spaces()[0], form->function_spaces()[1]};
    }
  }
  return spaces;
}

/// Create a sparsity pattern for a given form. The pattern is not
/// finalised, i.e. the caller is responsible for calling
/// SparsityPattern::assemble.
/// @param[in] a A bilinear form
/// @return The corresponding sparsity pattern
template <typename T>
la::SparsityPattern create_sparsity_pattern(const Form<T>& a)
{
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear form");
  }

  // Get dof maps and mesh
  std::array<const std::reference_wrapper<const fem::DofMap>, 2> dofmaps{
      *a.function_spaces().at(0)->dofmap(),
      *a.function_spaces().at(1)->dofmap()};
  std::shared_ptr mesh = a.mesh();
  assert(mesh);

  const std::set<IntegralType> types = a.integral_types();
  if (types.find(IntegralType::interior_facet) != types.end()
      or types.find(IntegralType::exterior_facet) != types.end())
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    const int tdim = mesh->topology().dim();
    mesh->topology_mutable().create_entities(tdim - 1);
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
  }

  return create_sparsity_pattern(mesh->topology(), dofmaps, types);
}

/// Create a sparsity pattern for a given form. The pattern is not
/// finalised, i.e. the caller is responsible for calling
/// SparsityPattern::assemble.
la::SparsityPattern create_sparsity_pattern(
    const mesh::Topology& topology,
    const std::array<const std::reference_wrapper<const fem::DofMap>, 2>&
        dofmaps,
    const std::set<IntegralType>& integrals);

/// Create an ElementDofLayout from a ufc_dofmap
ElementDofLayout create_element_dof_layout(const ufc_dofmap& dofmap,
                                           const mesh::CellType cell_type,
                                           const std::vector<int>& parent_map
                                           = {});

/// Create dof map on mesh from a ufc_dofmap
/// @param[in] comm MPI communicator
/// @param[in] dofmap The ufc_dofmap
/// @param[in] topology The mesh topology
DofMap create_dofmap(MPI_Comm comm, const ufc_dofmap& dofmap,
                     mesh::Topology& topology);

/// Get the name of each coefficient in a UFC form
/// @param[in] ufc_form The UFC form
/// return The name of each coefficient
std::vector<std::string> get_coefficient_names(const ufc_form& ufc_form);

/// Get the name of each constant in a UFC form
/// @param[in] ufc_form The UFC form
/// return The name of each constant
std::vector<std::string> get_constant_names(const ufc_form& ufc_form);

/// Create a Form from UFC input
/// @param[in] ufc_form The UFC form
/// @param[in] spaces Vector of function spaces
/// @param[in] coefficients Coefficient fields in the form
/// @param[in] constants Spatial constants in the form
/// @param[in] subdomains Subdomain markers
/// @param[in] mesh The mesh of the domain
template <typename T>
Form<T> create_form(
    const ufc_form& ufc_form,
    const std::vector<std::shared_ptr<const fem::FunctionSpace>>& spaces,
    const std::vector<std::shared_ptr<const fem::Function<T>>>& coefficients,
    const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants,
    const std::map<IntegralType, const mesh::MeshTags<int>*>& subdomains,
    const std::shared_ptr<const mesh::Mesh>& mesh = nullptr)
{
  if (ufc_form.rank != (int)spaces.size())
    throw std::runtime_error("Wrong number of argument spaces for Form.");
  if (ufc_form.num_coefficients != (int)coefficients.size())
  {
    throw std::runtime_error(
        "Mismatch between number of expected and provided Form coefficients.");
  }
  if (ufc_form.num_constants != (int)constants.size())
  {
    throw std::runtime_error(
        "Mismatch between number of expected and provided Form constants.");
  }

  // Check argument function spaces
#ifdef DEBUG
  for (std::size_t i = 0; i < spaces.size(); ++i)
  {
    assert(spaces[i]->element());
    std::unique_ptr<ufc_finite_element, decltype(free)*> ufc_element(
        ufc_form.create_finite_element(i), free);
    assert(ufc_element);
    if (std::string(ufc_element->signature)
        != spaces[i]->element()->signature())
    {
      throw std::runtime_error(
          "Cannot create form. Wrong type of function space for argument.");
    }
  }
#endif

  // Get list of integral IDs, and load tabulate tensor into memory for
  // each
  using kern
      = std::function<void(T*, const T*, const T*, const double*, const int*,
                           const std::uint8_t*, const std::uint32_t)>;
  std::map<IntegralType, std::pair<std::vector<std::pair<int, kern>>,
                                   const mesh::MeshTags<int>*>>
      integral_data;

  bool needs_permutation_data = false;

  // Attach cell kernels
  std::vector<int> cell_integral_ids(ufc_form.num_cell_integrals);
  ufc_form.get_cell_integral_ids(cell_integral_ids.data());
  for (int id : cell_integral_ids)
  {
    ufc_integral* integral = ufc_form.create_cell_integral(id);
    assert(integral);
    if (integral->needs_permutation_data)
      needs_permutation_data = true;
    integral_data[IntegralType::cell].first.emplace_back(
        id, integral->tabulate_tensor);
    std::free(integral);
  }

  // Attach cell subdomain data
  if (auto it = subdomains.find(IntegralType::cell);
      it != subdomains.end() and !cell_integral_ids.empty())
  {
    integral_data[IntegralType::cell].second = it->second;
  }

  // FIXME: Can facets be handled better?

  // Create facets, if required
  if (ufc_form.num_exterior_facet_integrals > 0
      or ufc_form.num_interior_facet_integrals > 0)
  {
    if (!spaces.empty())
    {
      auto mesh = spaces[0]->mesh();
      const int tdim = mesh->topology().dim();
      spaces[0]->mesh()->topology_mutable().create_entities(tdim - 1);
    }
  }

  // Attach exterior facet kernels
  std::vector<int> exterior_facet_integral_ids(
      ufc_form.num_exterior_facet_integrals);
  ufc_form.get_exterior_facet_integral_ids(exterior_facet_integral_ids.data());
  for (int id : exterior_facet_integral_ids)
  {
    ufc_integral* integral = ufc_form.create_exterior_facet_integral(id);
    assert(integral);
    if (integral->needs_permutation_data)
      needs_permutation_data = true;
    integral_data[IntegralType::exterior_facet].first.emplace_back(
        id, integral->tabulate_tensor);
    std::free(integral);
  }

  // Attach exterior facet subdomain data
  if (auto it = subdomains.find(IntegralType::exterior_facet);
      it != subdomains.end() and !exterior_facet_integral_ids.empty())
  {
    integral_data[IntegralType::exterior_facet].second = it->second;
  }

  // Attach interior facet kernels
  std::vector<int> interior_facet_integral_ids(
      ufc_form.num_interior_facet_integrals);
  ufc_form.get_interior_facet_integral_ids(interior_facet_integral_ids.data());
  for (int id : interior_facet_integral_ids)
  {
    ufc_integral* integral = ufc_form.create_interior_facet_integral(id);
    assert(integral);
    if (integral->needs_permutation_data)
      needs_permutation_data = true;
    integral_data[IntegralType::interior_facet].first.emplace_back(
        id, integral->tabulate_tensor);
    std::free(integral);
  }

  // Attach interior facet subdomain data
  if (auto it = subdomains.find(IntegralType::interior_facet);
      it != subdomains.end() and !interior_facet_integral_ids.empty())
  {
    integral_data[IntegralType::interior_facet].second = it->second;
  }

  // Vertex integrals: not currently working
  std::vector<int> vertex_integral_ids(ufc_form.num_vertex_integrals);
  ufc_form.get_vertex_integral_ids(vertex_integral_ids.data());
  if (!vertex_integral_ids.empty())
  {
    throw std::runtime_error(
        "Vertex integrals not supported. Under development.");
  }

  return fem::Form(spaces, integral_data, coefficients, constants,
                   needs_permutation_data, mesh);
}

/// Create a Form from UFC input
/// @param[in] ufc_form The UFC form
/// @param[in] spaces The function spaces for the Form arguments
/// @param[in] coefficients Coefficient fields in the form (by name)
/// @param[in] constants Spatial constants in the form (by name)
/// @param[in] subdomains Subdomain makers
/// @param[in] mesh The mesh of the domain. This is required if the form
/// has no arguments, e.g. a functional.
/// @return A Form
template <typename T>
Form<T> create_form(
    const ufc_form& ufc_form,
    const std::vector<std::shared_ptr<const fem::FunctionSpace>>& spaces,
    const std::map<std::string, std::shared_ptr<const fem::Function<T>>>&
        coefficients,
    const std::map<std::string, std::shared_ptr<const fem::Constant<T>>>&
        constants,
    const std::map<IntegralType, const mesh::MeshTags<int>*>& subdomains,
    const std::shared_ptr<const mesh::Mesh>& mesh = nullptr)
{
  // Place coefficients in appropriate order
  std::vector<std::shared_ptr<const fem::Function<T>>> coeff_map;
  for (const std::string& name : get_coefficient_names(ufc_form))
  {
    if (auto it = coefficients.find(name); it != coefficients.end())
      coeff_map.push_back(it->second);
    else
    {
      throw std::runtime_error("Form coefficient \"" + name
                               + "\" not provided.");
    }
  }

  // Place constants in appropriate order
  std::vector<std::shared_ptr<const fem::Constant<T>>> const_map;
  for (const std::string& name : get_constant_names(ufc_form))
  {
    if (auto it = constants.find(name); it != constants.end())
      const_map.push_back(it->second);
    else
    {
      throw std::runtime_error("Form constant \"" + name + "\" not provided.");
    }
  }

  return create_form(ufc_form, spaces, coeff_map, const_map, subdomains, mesh);
}

/// Create a Form using a factory function that returns a pointer to a
/// ufc_form.
/// @param[in] fptr pointer to a function returning a pointer to
/// ufc_form
/// @param[in] spaces The function spaces for the Form arguments
/// @param[in] coefficients Coefficient fields in the form (by name)
/// @param[in] constants Spatial constants in the form (by name)
/// @param[in] subdomains Subdomain markers
/// @param[in] mesh The mesh of the domain. This is required if the form
/// has no arguments, e.g. a functional.
/// @return A Form
template <typename T>
std::shared_ptr<Form<T>> create_form(
    ufc_form* (*fptr)(),
    const std::vector<std::shared_ptr<const fem::FunctionSpace>>& spaces,
    const std::map<std::string, std::shared_ptr<const fem::Function<T>>>&
        coefficients,
    const std::map<std::string, std::shared_ptr<const fem::Constant<T>>>&
        constants,
    const std::map<IntegralType, const mesh::MeshTags<int>*>& subdomains,
    const std::shared_ptr<const mesh::Mesh>& mesh = nullptr)
{
  ufc_form* form = fptr();
  auto L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
      *form, spaces, coefficients, constants, subdomains, mesh));
  std::free(form);
  return L;
}

/// Create a CoordinateElement from ufc
/// @param[in] ufc_cmap UFC coordinate mapping
/// @return A DOLFINX coordinate map
fem::CoordinateElement
create_coordinate_map(const ufc_coordinate_mapping& ufc_cmap);

/// Create a CoordinateElement from ufc
/// @param[in] fptr Function Pointer to a ufc_function_coordinate_map
///   function
/// @return A DOLFINX coordinate map
fem::CoordinateElement create_coordinate_map(ufc_coordinate_mapping* (*fptr)());

/// Create FunctionSpace from UFC
/// @param[in] fptr Function Pointer to a ufc_function_space_create
///   function
/// @param[in] function_name Name of a function whose function space to
///   create. Function name is the name of Python variable for
///   ufl.Coefficient, ufl.TrialFunction or ufl.TestFunction as defined
///   in the UFL file.
/// @param[in] mesh Mesh
/// @return The created FunctionSpace
std::shared_ptr<fem::FunctionSpace>
create_functionspace(ufc_function_space* (*fptr)(const char*),
                     const std::string function_name,
                     std::shared_ptr<mesh::Mesh> mesh);

// NOTE: This is subject to change
/// Pack coefficients of u of generic type U ready for assembly
template <typename U>
common::array2d<typename U::scalar_type> pack_coefficients(const U& u)
{
  using T = typename U::scalar_type;

  // Get form coefficient offsets and dofmaps
  const std::vector<std::shared_ptr<const fem::Function<T>>> coefficients
      = u.coefficients();
  const std::vector<int> offsets = u.coefficient_offsets();
  std::vector<const fem::DofMap*> dofmaps(coefficients.size());
  std::vector<int> bs(coefficients.size());
  std::vector<std::reference_wrapper<const std::vector<T>>> v;
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    dofmaps[i] = coefficients[i]->function_space()->dofmap().get();
    bs[i] = dofmaps[i]->bs();
    v.push_back(coefficients[i]->x()->array());
  }

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = u.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells
      = mesh->topology().index_map(tdim)->size_local()
        + mesh->topology().index_map(tdim)->num_ghosts();

  // Copy data into coefficient array
  common::array2d<T> c(num_cells, offsets.back());
  if (coefficients.size() > 0)
  {
    for (int cell = 0; cell < num_cells; ++cell)
    {
      for (std::size_t coeff = 0; coeff < dofmaps.size(); ++coeff)
      {
        tcb::span<const std::int32_t> dofs = dofmaps[coeff]->cell_dofs(cell);
        const std::vector<T>& _v = v[coeff];
        for (std::size_t i = 0; i < dofs.size(); ++i)
        {
          for (int k = 0; k < bs[coeff]; ++k)
          {
            c(cell, bs[coeff] * i + k + offsets[coeff])
                = _v[bs[coeff] * dofs[i] + k];
          }
        }
      }
    }
  }

  return c;
}

// NOTE: This is subject to change
/// Pack constants of u of generic type U ready for assembly
template <typename U>
std::vector<typename U::scalar_type> pack_constants(const U& u)
{
  using T = typename U::scalar_type;
  const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants
      = u.constants();

  // Calculate size of array needed to store packed constants
  std::int32_t size
      = std::accumulate(constants.begin(), constants.end(), 0,
                        [](std::int32_t sum, const auto& constant) {
                          return sum + constant->value.size();
                        });

  // Pack constants
  std::vector<T> constant_values(size);
  std::int32_t offset = 0;
  for (const auto& constant : constants)
  {
    const std::vector<T>& value = constant->value;
    for (std::size_t i = 0; i < value.size(); ++i)
      constant_values[offset + i] = value[i];
    offset += value.size();
  }

  return constant_values;
}

} // namespace fem
} // namespace dolfinx
