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
#include <dolfinx/function/Function.h>
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

namespace function
{
template <typename T>
class Constant;
template <typename T>
class Function;
class FunctionSpace;
} // namespace function

namespace mesh
{
class Mesh;
class Topology;
} // namespace mesh

namespace fem
{

/// Extract test (0) and trial (1) function spaces pairs for each
/// bilinear form for a rectangular array of forms
///
/// @param[in] a A rectangular block on bilinear forms
/// @return Rectangular array of the same shape as @p a with a pair of
///   function spaces in each array entry. If a form is null, then the
///   returned function space pair is (null, null).
template <typename T>
std::vector<
    std::vector<std::array<std::shared_ptr<const function::FunctionSpace>, 2>>>
extract_function_spaces(
    const Eigen::Ref<const Eigen::Array<const fem::Form<T>*, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>& a)
{
  std::vector<std::vector<
      std::array<std::shared_ptr<const function::FunctionSpace>, 2>>>
      spaces(a.rows(),
             std::vector<
                 std::array<std::shared_ptr<const function::FunctionSpace>, 2>>(
                 a.cols()));
  for (int i = 0; i < a.rows(); ++i)
    for (int j = 0; j < a.cols(); ++j)
      if (a(i, j))
        spaces[i][j] = {a(i, j)->function_space(0), a(i, j)->function_space(1)};
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
  std::array dofmaps{a.function_space(0)->dofmap().get(),
                     a.function_space(1)->dofmap().get()};
  std::shared_ptr mesh = a.mesh();
  assert(mesh);

  const std::set<IntegralType> types = a.integrals().types();
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
la::SparsityPattern
create_sparsity_pattern(const mesh::Topology& topology,
                        const std::array<const DofMap*, 2>& dofmaps,
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

/// Extract coefficients from a UFC form
template <typename T>
std::vector<
    std::tuple<int, std::string, std::shared_ptr<function::Function<T>>>>
get_coeffs_from_ufc_form(const ufc_form& ufc_form)
{
  std::vector<
      std::tuple<int, std::string, std::shared_ptr<function::Function<T>>>>
      coeffs;
  const char** names = ufc_form.coefficient_name_map();
  for (int i = 0; i < ufc_form.num_coefficients; ++i)
  {
    coeffs.emplace_back(ufc_form.original_coefficient_position(i), names[i],
                        nullptr);
  }
  return coeffs;
}

/// Extract coefficients from a UFC form
template <typename T>
std::vector<
    std::pair<std::string, std::shared_ptr<const function::Constant<T>>>>
get_constants_from_ufc_form(const ufc_form& ufc_form)
{
  std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant<T>>>>
      constants;
  const char** names = ufc_form.constant_name_map();
  for (int i = 0; i < ufc_form.num_constants; ++i)
    constants.emplace_back(names[i], nullptr);
  return constants;
}

/// Create a Form from UFC input
/// @param[in] ufc_form The UFC form
/// @param[in] spaces Vector of function spaces
template <typename T>
Form<T> create_form(
    const ufc_form& ufc_form,
    const std::vector<std::shared_ptr<const function::FunctionSpace>>& spaces)
{
  assert(ufc_form.rank == (int)spaces.size());

  // Check argument function spaces
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

  // Get list of integral IDs, and load tabulate tensor into memory for each
  bool needs_permutation_data = false;
  std::map<IntegralType,
           std::vector<std::pair<
               int, std::function<void(T*, const T*, const T*, const double*,
                                       const int*, const std::uint8_t*,
                                       const std::uint32_t)>>>>
      integral_data;

  std::vector<int> cell_integral_ids(ufc_form.num_cell_integrals);
  ufc_form.get_cell_integral_ids(cell_integral_ids.data());
  for (int id : cell_integral_ids)
  {
    ufc_integral* integral = ufc_form.create_cell_integral(id);
    assert(integral);
    if (integral->needs_permutation_data)
      needs_permutation_data = true;
    integral_data[IntegralType::cell].emplace_back(id,
                                                   integral->tabulate_tensor);
    std::free(integral);
  }

  // FIXME: Can this be handled better?
  // FIXME: Handle forms with no space
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

  std::vector<int> exterior_facet_integral_ids(
      ufc_form.num_exterior_facet_integrals);
  ufc_form.get_exterior_facet_integral_ids(exterior_facet_integral_ids.data());
  for (int id : exterior_facet_integral_ids)
  {
    ufc_integral* integral = ufc_form.create_exterior_facet_integral(id);
    assert(integral);
    if (integral->needs_permutation_data)
      needs_permutation_data = true;
    integral_data[IntegralType::exterior_facet].emplace_back(
        id, integral->tabulate_tensor);
    std::free(integral);
  }

  std::vector<int> interior_facet_integral_ids(
      ufc_form.num_interior_facet_integrals);
  ufc_form.get_interior_facet_integral_ids(interior_facet_integral_ids.data());
  for (int id : interior_facet_integral_ids)
  {
    ufc_integral* integral = ufc_form.create_interior_facet_integral(id);
    assert(integral);
    if (integral->needs_permutation_data)
      needs_permutation_data = true;
    integral_data[IntegralType::interior_facet].emplace_back(
        id, integral->tabulate_tensor);
    std::free(integral);
  }

  // Not currently working
  std::vector<int> vertex_integral_ids(ufc_form.num_vertex_integrals);
  ufc_form.get_vertex_integral_ids(vertex_integral_ids.data());
  if (vertex_integral_ids.size() > 0)
  {
    throw std::runtime_error(
        "Vertex integrals not supported. Under development.");
  }

  return fem::Form(
      spaces, FormIntegrals<T>(integral_data, needs_permutation_data),
      FormCoefficients<T>(fem::get_coeffs_from_ufc_form<T>(ufc_form)),
      fem::get_constants_from_ufc_form<T>(ufc_form));
}

/// Create a form from a form_create function returning a pointer to a
/// ufc_form, taking care of memory allocation
/// @param[in] fptr pointer to a function returning a pointer to
///    ufc_form
/// @param[in] spaces function spaces
/// @return Form
template <typename T>
std::shared_ptr<Form<T>> create_form(
    ufc_form* (*fptr)(),
    const std::vector<std::shared_ptr<const function::FunctionSpace>>& spaces)
{
  ufc_form* form = fptr();
  auto L = std::make_shared<fem::Form<T>>(
      dolfinx::fem::create_form<T>(*form, spaces));
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
std::shared_ptr<function::FunctionSpace>
create_functionspace(ufc_function_space* (*fptr)(const char*),
                     const std::string function_name,
                     std::shared_ptr<mesh::Mesh> mesh);

// NOTE: This is subject to change
/// Pack form coefficients ready for assembly
template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
pack_coefficients(const fem::Form<T>& form)
{
  // Get form coefficient offsets amd dofmaps
  const fem::FormCoefficients<T>& coefficients = form.coefficients();
  const std::vector<int>& offsets = coefficients.offsets();
  std::vector<const fem::DofMap*> dofmaps(coefficients.size());
  std::vector<Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>> v;
  for (int i = 0; i < coefficients.size(); ++i)
  {
    dofmaps[i] = coefficients.get(i)->function_space()->dofmap().get();
    v.emplace_back(coefficients.get(i)->x()->array());
  }

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = form.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells
      = mesh->topology().index_map(tdim)->size_local()
        + mesh->topology().index_map(tdim)->num_ghosts();

  // Copy data into coefficient array
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> c(
      num_cells, offsets.back());
  if (coefficients.size() > 0)
  {
    for (int cell = 0; cell < num_cells; ++cell)
    {
      for (std::size_t coeff = 0; coeff < dofmaps.size(); ++coeff)
      {
        auto dofs = dofmaps[coeff]->cell_dofs(cell);
        const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& _v
            = v[coeff];
        for (Eigen::Index k = 0; k < dofs.size(); ++k)
          c(cell, k + offsets[coeff]) = _v[dofs[k]];
      }
    }
  }

  return c;
}

// NOTE: This is subject to change
/// Pack form constants ready for assembly
template <typename T>
Eigen::Array<T, Eigen::Dynamic, 1> pack_constants(const fem::Form<T>& form)
{
  std::vector<T> constant_values;
  for (auto& constant : form.constants())
  {
    const std::vector<T>& array = constant.second->value;
    constant_values.insert(constant_values.end(), array.begin(), array.end());
  }

  return Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(
      constant_values.data(), constant_values.size(), 1);
}

} // namespace fem
} // namespace dolfinx
