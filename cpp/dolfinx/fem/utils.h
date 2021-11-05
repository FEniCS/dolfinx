// Copyright (C) 2013-2020 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CoordinateElement.h"
#include "DofMap.h"
#include "ElementDofLayout.h"
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/cell_types.h>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <ufc.h>
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::mesh
{
class Mesh;
class Topology;
} // namespace dolfinx::mesh

namespace dolfinx::fem
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
  std::array<std::reference_wrapper<const fem::DofMap>, 2> dofmaps{
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
    const std::array<std::reference_wrapper<const fem::DofMap>, 2>& dofmaps,
    const std::set<IntegralType>& integrals);

/// Create an ElementDofLayout from a ufc_dofmap
ElementDofLayout create_element_dof_layout(const ufc_dofmap& dofmap,
                                           const mesh::CellType cell_type,
                                           const std::vector<int>& parent_map
                                           = {});

/// Create a dof map on mesh from a ufc_dofmap
/// @param[in] comm MPI communicator
/// @param[in] dofmap The ufc_dofmap
/// @param[in] topology The mesh topology
/// @param[in] element The finite element
/// @param[in] reorder_fn The graph reordering function called on the
/// dofmap
DofMap
create_dofmap(MPI_Comm comm, const ufc_dofmap& dofmap, mesh::Topology& topology,
              const std::function<std::vector<int>(
                  const graph::AdjacencyList<std::int32_t>&)>& reorder_fn,
              std::shared_ptr<const dolfinx::fem::FiniteElement> element);

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
    ufc_finite_element* ufc_element = ufc_form.finite_elements[i];
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
  using kern = std::function<void(T*, const T*, const T*, const double*,
                                  const int*, const std::uint8_t*)>;
  std::map<IntegralType, std::pair<std::vector<std::pair<int, kern>>,
                                   const mesh::MeshTags<int>*>>
      integral_data;

  bool needs_facet_permutations = false;

  // Attach cell kernels
  std::vector<int> cell_integral_ids(ufc_form.integral_ids(cell),
                                     ufc_form.integral_ids(cell)
                                         + ufc_form.num_integrals(cell));
  for (int i = 0; i < ufc_form.num_integrals(cell); ++i)
  {
    ufc_integral* integral = ufc_form.integrals(cell)[i];
    assert(integral);

    kern k = nullptr;
    if constexpr (std::is_same<T, float>::value)
      k = integral->tabulate_tensor_float32;
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
      k = reinterpret_cast<void (*)(T*, const T*, const T*, const double*,
                                    const int*, const unsigned char*)>(
          integral->tabulate_tensor_complex64);
    }
    else if constexpr (std::is_same<T, double>::value)
      k = integral->tabulate_tensor_float64;
    else if constexpr (std::is_same<T, std::complex<double>>::value)
    {
      k = reinterpret_cast<void (*)(T*, const T*, const T*, const double*,
                                    const int*, const unsigned char*)>(
          integral->tabulate_tensor_complex128);
    }
    assert(k);

    integral_data[IntegralType::cell].first.emplace_back(cell_integral_ids[i],
                                                         k);
    if (integral->needs_facet_permutations)
      needs_facet_permutations = true;
  }

  // Attach cell subdomain data
  if (auto it = subdomains.find(IntegralType::cell);
      it != subdomains.end() and !cell_integral_ids.empty())
  {
    integral_data[IntegralType::cell].second = it->second;
  }

  // FIXME: Can facets be handled better?

  // Create facets, if required
  if (ufc_form.num_integrals(exterior_facet) > 0
      or ufc_form.num_integrals(interior_facet) > 0)
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
      ufc_form.integral_ids(exterior_facet),
      ufc_form.integral_ids(exterior_facet)
          + ufc_form.num_integrals(exterior_facet));
  for (int i = 0; i < ufc_form.num_integrals(exterior_facet); ++i)
  {
    ufc_integral* integral = ufc_form.integrals(exterior_facet)[i];
    assert(integral);

    kern k = nullptr;
    if constexpr (std::is_same<T, float>::value)
      k = integral->tabulate_tensor_float32;
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
      k = reinterpret_cast<void (*)(T*, const T*, const T*, const double*,
                                    const int*, const unsigned char*)>(
          integral->tabulate_tensor_complex64);
    }
    else if constexpr (std::is_same<T, double>::value)
      k = integral->tabulate_tensor_float64;
    else if constexpr (std::is_same<T, std::complex<double>>::value)
    {
      k = reinterpret_cast<void (*)(T*, const T*, const T*, const double*,
                                    const int*, const unsigned char*)>(
          integral->tabulate_tensor_complex128);
    }
    assert(k);

    integral_data[IntegralType::exterior_facet].first.emplace_back(
        exterior_facet_integral_ids[i], k);
    if (integral->needs_facet_permutations)
      needs_facet_permutations = true;
  }

  // Attach exterior facet subdomain data
  if (auto it = subdomains.find(IntegralType::exterior_facet);
      it != subdomains.end() and !exterior_facet_integral_ids.empty())
  {
    integral_data[IntegralType::exterior_facet].second = it->second;
  }

  // Attach interior facet kernels
  std::vector<int> interior_facet_integral_ids(
      ufc_form.integral_ids(interior_facet),
      ufc_form.integral_ids(interior_facet)
          + ufc_form.num_integrals(interior_facet));
  for (int i = 0; i < ufc_form.num_integrals(interior_facet); ++i)
  {
    ufc_integral* integral = ufc_form.integrals(interior_facet)[i];
    assert(integral);

    kern k = nullptr;
    if constexpr (std::is_same<T, float>::value)
      k = integral->tabulate_tensor_float32;
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
      k = reinterpret_cast<void (*)(T*, const T*, const T*, const double*,
                                    const int*, const unsigned char*)>(
          integral->tabulate_tensor_complex64);
    }
    else if constexpr (std::is_same<T, double>::value)
      k = integral->tabulate_tensor_float64;
    else if constexpr (std::is_same<T, std::complex<double>>::value)
    {
      k = reinterpret_cast<void (*)(T*, const T*, const T*, const double*,
                                    const int*, const unsigned char*)>(
          integral->tabulate_tensor_complex128);
    }
    assert(k);

    integral_data[IntegralType::interior_facet].first.emplace_back(
        interior_facet_integral_ids[i], k);
    if (integral->needs_facet_permutations)
      needs_facet_permutations = true;
  }

  // Attach interior facet subdomain data
  if (auto it = subdomains.find(IntegralType::interior_facet);
      it != subdomains.end() and !interior_facet_integral_ids.empty())
  {
    integral_data[IntegralType::interior_facet].second = it->second;
  }

  return fem::Form<T>(spaces, integral_data, coefficients, constants,
                      needs_facet_permutations, mesh);
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
      throw std::runtime_error("Form constant \"" + name + "\" not provided.");
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
Form<T> create_form(
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
  Form<T> L = fem::create_form<T>(*form, spaces, coefficients, constants,
                                  subdomains, mesh);
  std::free(form);
  return L;
}

/// Create a FunctionSpace from UFC data
///
/// @param[in] fptr Function Pointer to a ufc_function_space_create
/// function
/// @param[in] function_name Name of a function whose function space to
/// create. Function name is the name of Python variable for
/// ufl.Coefficient, ufl.TrialFunction or ufl.TestFunction as defined in
/// the UFL file.
/// @param[in] mesh Mesh
/// @param[in] reorder_fn The graph reordering function called on the
/// dofmap
/// @return The created function space
fem::FunctionSpace create_functionspace(
    ufc_function_space* (*fptr)(const char*), const std::string& function_name,
    std::shared_ptr<mesh::Mesh> mesh,
    const std::function<
        std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>& reorder_fn
    = nullptr);

namespace impl
{
// Pack a single coefficient
template <typename T, int _bs = -1>
void pack_coefficient(
    const xtl::span<T>& c, int cstride, const xtl::span<const T>& v,
    const xtl::span<const std::uint32_t>& cell_info, const fem::DofMap& dofmap,
    std::int32_t num_cells, std::int32_t offset, int space_dim,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& transformation)
{
  const int bs = dofmap.bs();
  assert(_bs < 0 or _bs == bs);
  for (std::int32_t cell = 0; cell < num_cells; ++cell)
  {
    auto dofs = dofmap.cell_dofs(cell);
    auto cell_coeff = c.subspan(cell * cstride + offset, space_dim);
    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
      if constexpr (_bs < 0)
      {
        const int pos_c = bs * i;
        const int pos_v = bs * dofs[i];
        for (int k = 0; k < bs; ++k)
          cell_coeff[pos_c + k] = v[pos_v + k];
      }
      else
      {
        const int pos_c = _bs * i;
        const int pos_v = _bs * dofs[i];
        for (int k = 0; k < _bs; ++k)
          cell_coeff[pos_c + k] = v[pos_v + k];
      }
    }

    transformation(cell_coeff, cell_info, cell, 1);
  }
}
} // namespace impl

// NOTE: This is subject to change
/// Pack coefficients of u of generic type U ready for assembly
template <typename U>
std::pair<std::vector<typename U::scalar_type>, int>
pack_coefficients(const U& u)
{
  using T = typename U::scalar_type;

  // Get form coefficient offsets and dofmaps
  const std::vector<std::shared_ptr<const fem::Function<T>>> coefficients
      = u.coefficients();
  const std::vector<int> offsets = u.coefficient_offsets();
  std::vector<const fem::DofMap*> dofmaps(coefficients.size());
  std::vector<const fem::FiniteElement*> elements(coefficients.size());
  std::vector<xtl::span<const T>> v;
  v.reserve(coefficients.size());
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    elements[i] = coefficients[i]->function_space()->element().get();
    dofmaps[i] = coefficients[i]->function_space()->dofmap().get();
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
  std::vector<T> c(num_cells * offsets.back());
  const int cstride = offsets.back();
  if (!coefficients.empty())
  {
    bool needs_dof_transformations = false;
    for (std::size_t coeff = 0; coeff < dofmaps.size(); ++coeff)
    {
      if (elements[coeff]->needs_dof_transformations())
      {
        needs_dof_transformations = true;
        mesh->topology_mutable().create_entity_permutations();
      }
    }

    // Iterate over coefficients
    xtl::span<const std::uint32_t> cell_info;
    if (needs_dof_transformations)
      cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
    for (std::size_t coeff = 0; coeff < dofmaps.size(); ++coeff)
    {
      const std::function<void(const xtl::span<T>&,
                               const xtl::span<const std::uint32_t>&,
                               std::int32_t, int)>
          transformation
          = elements[coeff]->get_dof_transformation_function<T>(false, true);
      if (int bs = dofmaps[coeff]->bs(); bs == 1)
      {
        impl::pack_coefficient<T, 1>(
            xtl::span<T>(c), cstride, v[coeff], cell_info, *dofmaps[coeff],
            num_cells, offsets[coeff], elements[coeff]->space_dimension(),
            transformation);
      }
      else if (bs == 2)
      {
        impl::pack_coefficient<T, 2>(
            xtl::span<T>(c), cstride, v[coeff], cell_info, *dofmaps[coeff],
            num_cells, offsets[coeff], elements[coeff]->space_dimension(),
            transformation);
      }
      else if (bs == 3)
      {
        impl::pack_coefficient<T, 3>(
            xtl::span<T>(c), cstride, v[coeff], cell_info, *dofmaps[coeff],
            num_cells, offsets[coeff], elements[coeff]->space_dimension(),
            transformation);
      }
      else
      {
        impl::pack_coefficient<T>(xtl::span<T>(c), cstride, v[coeff], cell_info,
                                  *dofmaps[coeff], num_cells, offsets[coeff],
                                  elements[coeff]->space_dimension(),
                                  transformation);
      }
    }
  }

  return {std::move(c), cstride};
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
  std::int32_t size = std::accumulate(constants.begin(), constants.end(), 0,
                                      [](std::int32_t sum, const auto& constant)
                                      { return sum + constant->value.size(); });

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

} // namespace dolfinx::fem
