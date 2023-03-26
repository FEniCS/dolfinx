// Copyright (C) 2013-2020 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "CoordinateElement.h"
#include "DofMap.h"
#include "ElementDofLayout.h"
#include "Expression.h"
#include "Form.h"
#include "Function.h"
#include "sparsitybuild.h"
#include <array>
#include <concepts>
#include <dolfinx/common/types.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <functional>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <ufcx.h>
#include <utility>
#include <vector>

/// @file utils.h
/// @brief Functions supporting finite element method operations

namespace basix
{
class FiniteElement;
}

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::mesh
{
class Topology;
} // namespace dolfinx::mesh

namespace dolfinx::fem
{
template <typename T>
class FunctionSpace;

namespace impl
{
/// Helper function to get an array of of (cell, local_facet) pairs
/// corresponding to a given facet index.
/// @param[in] f Facet index
/// @param[in] cells List of cells incident to the facet
/// @param[in] c_to_f Cell to facet connectivity
/// @return Vector of (cell, local_facet) pairs
template <int num_cells>
std::array<std::int32_t, 2 * num_cells>
get_cell_facet_pairs(std::int32_t f, const std::span<const std::int32_t>& cells,
                     const graph::AdjacencyList<std::int32_t>& c_to_f)
{
  // Loop over cells sharing facet
  assert(cells.size() == num_cells);
  std::array<std::int32_t, 2 * num_cells> cell_local_facet_pairs;
  for (int c = 0; c < num_cells; ++c)
  {
    // Get local index of facet with respect to the cell
    std::int32_t cell = cells[c];
    auto cell_facets = c_to_f.links(cell);
    auto facet_it = std::find(cell_facets.begin(), cell_facets.end(), f);
    assert(facet_it != cell_facets.end());
    int local_f = std::distance(cell_facets.begin(), facet_it);
    cell_local_facet_pairs[2 * c] = cell;
    cell_local_facet_pairs[2 * c + 1] = local_f;
  }

  return cell_local_facet_pairs;
}

} // namespace impl

/// @brief Given an integral type and MeshTags, compute the entities
/// that should be integrated over.
///
/// This function returns as list `[(id, entities)]`, where `entities`
/// are the entities in `meshtags` with tag value `id`. For cell
/// integrals `entities` are the cell indices. For exterior facet
/// integrals, `entities` is a list of `(cell_index, local_facet_index)`
/// pairs. For interior facet integrals, `entities` is a list of
/// `(cell_index0, local_facet_index0, cell_index1,
/// local_facet_index1)`.
///
/// @note Owned mesh entities only are returned. Ghost entities are not
/// included.
///
/// @param[in] integral_type Integral type
/// @param[in] topology Mesh topology
/// @param[in] entities List of tagged mesh entities
/// @param[in] dim Topological dimension of tagged entities
/// @param[in] values Value associated with each entity
/// @return A list of (integral id, entities) pairs
/// @pre The topological dimension of the integral entity type and the
/// topological dimension of `meshtags` must be equal.
std::vector<std::pair<int, std::vector<std::int32_t>>>
compute_integration_domains(IntegralType integral_type,
                            mesh::Topology& topology,
                            std::span<const std::int32_t> entities, int dim,
                            std::span<const int> values);

/// @brief Finite element cell kernel concept.
///
/// Kernel functions that can be passed to an assembler for execution
/// must satisfy this concept.
template <class U, class T>
concept FEkernel = std::is_invocable_v<U, T*, const T*, const T*,
                                       const scalar_value_type_t<T>*,
                                       const int*, const std::uint8_t*>;

/// @brief Extract test (0) and trial (1) function spaces pairs for each
/// bilinear form for a rectangular array of forms.
///
/// @param[in] a A rectangular block on bilinear forms
/// @return Rectangular array of the same shape as `a` with a pair of
/// function spaces in each array entry. If a form is null, then the
/// returned function space pair is (null, null).
template <typename T, typename U>
std::vector<std::vector<std::array<std::shared_ptr<const FunctionSpace<U>>, 2>>>
extract_function_spaces(const std::vector<std::vector<const Form<T, U>*>>& a)
{
  std::vector<
      std::vector<std::array<std::shared_ptr<const FunctionSpace<U>>, 2>>>
      spaces(
          a.size(),
          std::vector<std::array<std::shared_ptr<const FunctionSpace<U>>, 2>>(
              a[0].size()));
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < a[i].size(); ++j)
    {
      if (const Form<T, U>* form = a[i][j]; form)
        spaces[i][j] = {form->function_spaces()[0], form->function_spaces()[1]};
    }
  }
  return spaces;
}

/// @brief Create a sparsity pattern for a given form.
/// @note The pattern is not finalised, i.e. the caller is responsible
/// for calling SparsityPattern::assemble.
la::SparsityPattern create_sparsity_pattern(
    const mesh::Topology& topology,
    const std::array<std::reference_wrapper<const DofMap>, 2>& dofmaps,
    const std::set<IntegralType>& integrals);

/// @brief Create a sparsity pattern for a given form.
/// @note The pattern is not finalised, i.e. the caller is responsible
/// for calling SparsityPattern::assemble.
/// @param[in] a A bilinear form
/// @return The corresponding sparsity pattern
template <typename T, typename U>
la::SparsityPattern create_sparsity_pattern(const Form<T, U>& a)
{
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear form");
  }

  // Get dof maps and mesh
  std::array<std::reference_wrapper<const DofMap>, 2> dofmaps{
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

  common::Timer t0("Build sparsity");

  // Get common::IndexMaps for each dimension
  const std::array index_maps{dofmaps[0].get().index_map,
                              dofmaps[1].get().index_map};
  const std::array bs
      = {dofmaps[0].get().index_map_bs(), dofmaps[1].get().index_map_bs()};

  // Create and build sparsity pattern
  la::SparsityPattern pattern(mesh->comm(), index_maps, bs);
  for (auto type : types)
  {
    std::vector<int> ids = a.integral_ids(type);
    switch (type)
    {
    case IntegralType::cell:
      for (int id : ids)
      {
        const std::vector<std::int32_t>& cells = a.cell_domains(id);
        sparsitybuild::cells(pattern, cells, {{dofmaps[0], dofmaps[1]}});
      }
      break;
    case IntegralType::interior_facet:
      for (int id : ids)
      {
        const std::vector<std::int32_t>& facets = a.interior_facet_domains(id);
        sparsitybuild::interior_facets(pattern, facets,
                                       {{dofmaps[0], dofmaps[1]}});
      }
      break;
    case IntegralType::exterior_facet:
      for (int id : ids)
      {
        const std::vector<std::int32_t>& facets = a.exterior_facet_domains(id);
        sparsitybuild::exterior_facets(pattern, facets,
                                       {{dofmaps[0], dofmaps[1]}});
      }
      break;
    default:
      throw std::runtime_error("Unsupported integral type");
    }
  }

  t0.stop();

  return pattern;
}

/// Create an ElementDofLayout from a ufcx_dofmap
ElementDofLayout create_element_dof_layout(const ufcx_dofmap& dofmap,
                                           const mesh::CellType cell_type,
                                           const std::vector<int>& parent_map
                                           = {});

/// @brief Create a dof map on mesh
/// @param[in] comm MPI communicator
/// @param[in] layout The dof layout on an element
/// @param[in] topology The mesh topology
/// @param[in] element The finite element
/// @param[in] reorder_fn The graph reordering function called on the
/// dofmap
/// @return A new dof map
DofMap
create_dofmap(MPI_Comm comm, const ElementDofLayout& layout,
              mesh::Topology& topology,
              const std::function<std::vector<int>(
                  const graph::AdjacencyList<std::int32_t>&)>& reorder_fn,
              const FiniteElement& element);

/// Get the name of each coefficient in a UFC form
/// @param[in] ufcx_form The UFC form
/// @return The name of each coefficient
std::vector<std::string> get_coefficient_names(const ufcx_form& ufcx_form);

/// @brief Get the name of each constant in a UFC form
/// @param[in] ufcx_form The UFC form
/// @return The name of each constant
std::vector<std::string> get_constant_names(const ufcx_form& ufcx_form);

/// @brief Create a Form from UFC input
/// @param[in] ufcx_form The UFC form
/// @param[in] spaces Vector of function spaces
/// @param[in] coefficients Coefficient fields in the form
/// @param[in] constants Spatial constants in the form
/// @param[in] subdomains Subdomain markers
/// @pre Each value in `subdomains` must be sorted by domain id
/// @param[in] mesh The mesh of the domain
template <typename T, typename U = dolfinx::scalar_value_type_t<T>>
Form<T, U> create_form(
    const ufcx_form& ufcx_form,
    const std::vector<std::shared_ptr<const FunctionSpace<U>>>& spaces,
    const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients,
    const std::vector<std::shared_ptr<const Constant<T>>>& constants,
    const std::map<
        IntegralType,
        std::vector<std::pair<std::int32_t, std::vector<std::int32_t>>>>&
        subdomains,
    std::shared_ptr<const mesh::Mesh<U>> mesh = nullptr)
{
  if (ufcx_form.rank != (int)spaces.size())
    throw std::runtime_error("Wrong number of argument spaces for Form.");
  if (ufcx_form.num_coefficients != (int)coefficients.size())
  {
    throw std::runtime_error(
        "Mismatch between number of expected and provided Form coefficients.");
  }
  if (ufcx_form.num_constants != (int)constants.size())
  {
    throw std::runtime_error(
        "Mismatch between number of expected and provided Form constants.");
  }

  // Check argument function spaces
#ifndef NDEBUG
  for (std::size_t i = 0; i < spaces.size(); ++i)
  {
    assert(spaces[i]->element());
    ufcx_finite_element* ufcx_element = ufcx_form.finite_elements[i];
    assert(ufcx_element);
    if (std::string(ufcx_element->signature)
        != spaces[i]->element()->signature())
    {
      throw std::runtime_error(
          "Cannot create form. Wrong type of function space for argument.");
    }
  }
#endif

  // Extract mesh from FunctionSpace, and check they are the same
  if (!mesh and !spaces.empty())
    mesh = spaces[0]->mesh();
  for (auto& V : spaces)
  {
    if (mesh != V->mesh())
      throw std::runtime_error("Incompatible mesh");
  }
  if (!mesh)
    throw std::runtime_error("No mesh could be associated with the Form.");

  const mesh::Topology& topology = mesh->topology();
  const int tdim = topology.dim();

  // Create facets, if required
  if (ufcx_form.num_integrals(exterior_facet) > 0
      or ufcx_form.num_integrals(interior_facet) > 0)
  {
    mesh->topology_mutable().create_entities(tdim - 1);
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
  }

  // Get list of integral IDs, and load tabulate tensor into memory for
  // each
  using kern = std::function<void(
      T*, const T*, const T*, const typename scalar_value_type<T>::value_type*,
      const int*, const std::uint8_t*)>;
  std::map<IntegralType,
           std::vector<std::tuple<int, kern, std::vector<std::int32_t>>>>
      integral_data;

  bool needs_facet_permutations = false;

  // Attach cell kernels
  {
    std::span<const int> ids(ufcx_form.integral_ids(cell),
                             ufcx_form.num_integrals(cell));
    auto itg = integral_data.insert({IntegralType::cell, {}});
    auto sd = subdomains.find(IntegralType::cell);
    for (int i = 0; i < ufcx_form.num_integrals(cell); ++i)
    {
      const int id = ids[i];
      ufcx_integral* integral = ufcx_form.integrals(cell)[i];
      assert(integral);

      kern k = nullptr;
      if constexpr (std::is_same_v<T, float>)
        k = integral->tabulate_tensor_float32;
      else if constexpr (std::is_same_v<T, std::complex<float>>)
      {
        k = reinterpret_cast<void (*)(
            T*, const T*, const T*,
            const typename scalar_value_type<T>::value_type*, const int*,
            const unsigned char*)>(integral->tabulate_tensor_complex64);
      }
      else if constexpr (std::is_same_v<T, double>)
        k = integral->tabulate_tensor_float64;
      else if constexpr (std::is_same_v<T, std::complex<double>>)
      {
        k = reinterpret_cast<void (*)(
            T*, const T*, const T*,
            const typename scalar_value_type<T>::value_type*, const int*,
            const unsigned char*)>(integral->tabulate_tensor_complex128);
      }
      assert(k);

      // Build list of entities to assembler over
      if (id == -1)
      {
        // Default kernel, operates on all (owned) cells
        assert(topology.index_map(tdim));
        std::vector<std::int32_t> e;
        e.resize(topology.index_map(tdim)->size_local(), 0);
        std::iota(e.begin(), e.end(), 0);
        itg.first->second.emplace_back(id, k, std::move(e));
      }
      else if (sd != subdomains.end())
      {
        // NOTE: This requires that pairs are sorted
        auto it = std::lower_bound(sd->second.begin(), sd->second.end(), id,
                                   [](auto& pair, auto val)
                                   { return pair.first < val; });
        if (it != sd->second.end() and it->first == id)
          itg.first->second.emplace_back(id, k, it->second);
      }

      if (integral->needs_facet_permutations)
        needs_facet_permutations = true;
    }
  }

  // Attach exterior facet kernels
  {
    std::span<const int> ids(ufcx_form.integral_ids(exterior_facet),
                             ufcx_form.num_integrals(exterior_facet));
    auto itg = integral_data.insert({IntegralType::exterior_facet, {}});
    auto sd = subdomains.find(IntegralType::exterior_facet);
    for (int i = 0; i < ufcx_form.num_integrals(exterior_facet); ++i)
    {
      const int id = ids[i];
      ufcx_integral* integral = ufcx_form.integrals(exterior_facet)[i];
      assert(integral);

      kern k = nullptr;
      if constexpr (std::is_same_v<T, float>)
        k = integral->tabulate_tensor_float32;
      else if constexpr (std::is_same_v<T, std::complex<float>>)
      {
        k = reinterpret_cast<void (*)(
            T*, const T*, const T*,
            const typename scalar_value_type<T>::value_type*, const int*,
            const unsigned char*)>(integral->tabulate_tensor_complex64);
      }
      else if constexpr (std::is_same_v<T, double>)
        k = integral->tabulate_tensor_float64;
      else if constexpr (std::is_same_v<T, std::complex<double>>)
      {
        k = reinterpret_cast<void (*)(
            T*, const T*, const T*,
            const typename scalar_value_type<T>::value_type*, const int*,
            const unsigned char*)>(integral->tabulate_tensor_complex128);
      }
      assert(k);

      // Build list of entities to assembler over
      const std::vector bfacets = mesh::exterior_facet_indices(topology);
      auto f_to_c = topology.connectivity(tdim - 1, tdim);
      assert(f_to_c);
      auto c_to_f = topology.connectivity(tdim, tdim - 1);
      assert(c_to_f);
      if (id == -1)
      {
        // Default kernel, operates on all (owned) exterior facets
        std::vector<std::int32_t> e;
        e.reserve(2 * bfacets.size());
        for (std::int32_t f : bfacets)
        {
          // There will only be one pair for an exterior facet integral
          auto pair
              = impl::get_cell_facet_pairs<1>(f, f_to_c->links(f), *c_to_f);
          e.insert(e.end(), pair.begin(), pair.end());
        }
        itg.first->second.emplace_back(id, k, std::move(e));
      }
      else if (sd != subdomains.end())
      {
        // NOTE: This requires that pairs are sorted
        auto it = std::lower_bound(sd->second.begin(), sd->second.end(), id,
                                   [](auto& pair, auto val)
                                   { return pair.first < val; });
        if (it != sd->second.end() and it->first == id)
          itg.first->second.emplace_back(id, k, it->second);
      }

      if (integral->needs_facet_permutations)
        needs_facet_permutations = true;
    }
  }

  // Attach interior facet kernels
  {
    std::span<const int> ids(ufcx_form.integral_ids(interior_facet),
                             ufcx_form.num_integrals(interior_facet));
    auto itg = integral_data.insert({IntegralType::interior_facet, {}});
    auto sd = subdomains.find(IntegralType::interior_facet);
    for (int i = 0; i < ufcx_form.num_integrals(interior_facet); ++i)
    {
      const int id = ids[i];
      ufcx_integral* integral = ufcx_form.integrals(interior_facet)[i];
      assert(integral);

      kern k = nullptr;
      if constexpr (std::is_same_v<T, float>)
        k = integral->tabulate_tensor_float32;
      else if constexpr (std::is_same_v<T, std::complex<float>>)
      {
        k = reinterpret_cast<void (*)(
            T*, const T*, const T*,
            const typename scalar_value_type<T>::value_type*, const int*,
            const unsigned char*)>(integral->tabulate_tensor_complex64);
      }
      else if constexpr (std::is_same_v<T, double>)
        k = integral->tabulate_tensor_float64;
      else if constexpr (std::is_same_v<T, std::complex<double>>)
      {
        k = reinterpret_cast<void (*)(
            T*, const T*, const T*,
            const typename scalar_value_type<T>::value_type*, const int*,
            const unsigned char*)>(integral->tabulate_tensor_complex128);
      }
      assert(k);

      // Build list of entities to assembler over
      auto f_to_c = topology.connectivity(tdim - 1, tdim);
      assert(f_to_c);
      auto c_to_f = topology.connectivity(tdim, tdim - 1);
      assert(c_to_f);
      if (id == -1)
      {
        // Default kernel, operates on all (owned) interior facets
        std::vector<std::int32_t> e;
        assert(topology.index_map(tdim - 1));
        std::int32_t num_facets = topology.index_map(tdim - 1)->size_local();
        e.reserve(4 * num_facets);
        for (std::int32_t f = 0; f < num_facets; ++f)
        {
          if (f_to_c->num_links(f) == 2)
          {
            auto pairs
                = impl::get_cell_facet_pairs<2>(f, f_to_c->links(f), *c_to_f);
            e.insert(e.end(), pairs.begin(), pairs.end());
          }
        }
        itg.first->second.emplace_back(id, k, std::move(e));
      }
      else if (sd != subdomains.end())
      {
        auto it = std::lower_bound(sd->second.begin(), sd->second.end(), id,
                                   [](auto& pair, auto val)
                                   { return pair.first < val; });
        if (it != sd->second.end() and it->first == id)
          itg.first->second.emplace_back(id, k, it->second);
      }

      if (integral->needs_facet_permutations)
        needs_facet_permutations = true;
    }
  }

  std::map<IntegralType,
           std::vector<std::pair<std::int32_t, std::vector<std::int32_t>>>>
      sd;
  for (auto& [itg, data] : subdomains)
  {
    std::vector<std::pair<std::int32_t, std::vector<std::int32_t>>> x;
    for (auto& [id, idx] : data)
      x.emplace_back(id, std::vector(idx.data(), idx.data() + idx.size()));
    sd.insert({itg, std::move(x)});
  }

  return Form<T, U>(spaces, integral_data, coefficients, constants,
                    needs_facet_permutations, mesh);
}

/// @brief Create a Form from UFC input
/// @param[in] ufcx_form The UFC form
/// @param[in] spaces The function spaces for the Form arguments
/// @param[in] coefficients Coefficient fields in the form (by name)
/// @param[in] constants Spatial constants in the form (by name)
/// @param[in] subdomains Subdomain makers
/// @pre Each value in `subdomains` must be sorted by domain id
/// @param[in] mesh The mesh of the domain. This is required if the form
/// has no arguments, e.g. a functional
/// @return A Form
template <typename T, typename U = dolfinx::scalar_value_type_t<T>>
Form<T, U> create_form(
    const ufcx_form& ufcx_form,
    const std::vector<std::shared_ptr<const FunctionSpace<U>>>& spaces,
    const std::map<std::string, std::shared_ptr<const Function<T, U>>>&
        coefficients,
    const std::map<std::string, std::shared_ptr<const Constant<T>>>& constants,
    const std::map<
        IntegralType,
        std::vector<std::pair<std::int32_t, std::vector<std::int32_t>>>>&
        subdomains,
    std::shared_ptr<const mesh::Mesh<U>> mesh = nullptr)
{
  // Place coefficients in appropriate order
  std::vector<std::shared_ptr<const Function<T, U>>> coeff_map;
  for (const std::string& name : get_coefficient_names(ufcx_form))
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
  std::vector<std::shared_ptr<const Constant<T>>> const_map;
  for (const std::string& name : get_constant_names(ufcx_form))
  {
    if (auto it = constants.find(name); it != constants.end())
      const_map.push_back(it->second);
    else
      throw std::runtime_error("Form constant \"" + name + "\" not provided.");
  }

  return create_form(ufcx_form, spaces, coeff_map, const_map, subdomains, mesh);
}

/// @brief Create a Form using a factory function that returns a pointer
/// to a ufcx_form
/// @param[in] fptr pointer to a function returning a pointer to
/// ufcx_form
/// @param[in] spaces The function spaces for the Form arguments
/// @param[in] coefficients Coefficient fields in the form (by name)
/// @param[in] constants Spatial constants in the form (by name)
/// @param[in] subdomains Subdomain markers
/// @pre Each value in `subdomains` must be sorted by domain id
/// @param[in] mesh The mesh of the domain. This is required if the form
/// has no arguments, e.g. a functional.
/// @return A Form
template <typename T, typename U = dolfinx::scalar_value_type_t<T>>
Form<T, U> create_form(
    ufcx_form* (*fptr)(),
    const std::vector<std::shared_ptr<const FunctionSpace<U>>>& spaces,
    const std::map<std::string, std::shared_ptr<const Function<T, U>>>&
        coefficients,
    const std::map<std::string, std::shared_ptr<const Constant<T>>>& constants,
    const std::map<
        IntegralType,
        std::vector<std::pair<std::int32_t, std::vector<std::int32_t>>>>&
        subdomains,
    std::shared_ptr<const mesh::Mesh<U>> mesh = nullptr)
{
  ufcx_form* form = fptr();
  Form<T, U> L = create_form<T, U>(*form, spaces, coefficients, constants,
                                   subdomains, mesh);
  std::free(form);
  return L;
}

/// @brief Create a FunctionSpace from a Basix element
/// @param[in] mesh Mesh
/// @param[in] e Basix finite element
/// @param[in] bs The block size, e.g. 3 for a 'vector' Lagrange element
/// in 3D
/// @param[in] reorder_fn The graph reordering function to call on the
/// dofmap. If `nullptr`, the default re-ordering is used.
/// @return The created function space
template <typename T>
FunctionSpace<T>
create_functionspace(std::shared_ptr<mesh::Mesh<T>> mesh,
                     const basix::FiniteElement& e, int bs,
                     const std::function<std::vector<int>(
                         const graph::AdjacencyList<std::int32_t>&)>& reorder_fn
                     = nullptr)
{
  // Create a DOLFINx element
  auto _e = std::make_shared<FiniteElement>(e, bs);

  // Create UFC subdofmaps and compute offset
  assert(_e);
  const int num_sub_elements = _e->num_sub_elements();
  std::vector<ElementDofLayout> sub_doflayout;
  sub_doflayout.reserve(num_sub_elements);
  for (int i = 0; i < num_sub_elements; ++i)
  {
    auto sub_element = _e->extract_sub_element({i});
    std::vector<int> parent_map_sub(sub_element->space_dimension());
    for (std::size_t j = 0; j < parent_map_sub.size(); ++j)
      parent_map_sub[j] = i + bs * j;
    sub_doflayout.emplace_back(1, e.entity_dofs(), e.entity_closure_dofs(),
                               parent_map_sub, std::vector<ElementDofLayout>());
  }

  // Create a dofmap
  ElementDofLayout layout(bs, e.entity_dofs(), e.entity_closure_dofs(), {},
                          sub_doflayout);
  assert(mesh);
  auto dofmap = std::make_shared<const DofMap>(
      create_dofmap(mesh->comm(), layout, mesh->topology(), reorder_fn, *_e));

  return FunctionSpace(mesh, _e, dofmap);
}

/// Create a FunctionSpace from UFC data
///
/// @param[in] fptr Function Pointer to a ufcx_function_space_create
/// function
/// @param[in] function_name Name of a function whose function space to
/// create. Function name is the name of Python variable for
/// ufl.Coefficient, ufl.TrialFunction or ufl.TestFunction as defined in
/// the UFL file.
/// @param[in] mesh Mesh
/// @param[in] reorder_fn The graph reordering function to call on the
/// dofmap. If `nullptr`, the default re-ordering is used.
/// @return The created function space
template <typename T>
FunctionSpace<T>
create_functionspace(ufcx_function_space* (*fptr)(const char*),
                     const std::string& function_name,
                     std::shared_ptr<mesh::Mesh<T>> mesh,
                     const std::function<std::vector<int>(
                         const graph::AdjacencyList<std::int32_t>&)>& reorder_fn
                     = nullptr)
{
  ufcx_function_space* space = fptr(function_name.c_str());
  if (!space)
  {
    throw std::runtime_error(
        "Could not create UFC function space with function name "
        + function_name);
  }

  ufcx_finite_element* ufcx_element = space->finite_element;
  assert(ufcx_element);

  if (space->geometry_degree != mesh->geometry().cmap().degree()
      or static_cast<basix::cell::type>(space->geometry_basix_cell)
             != mesh::cell_type_to_basix_type(
                 mesh->geometry().cmap().cell_shape())
      or static_cast<basix::element::lagrange_variant>(
             space->geometry_basix_variant)
             != mesh->geometry().cmap().variant())
  {
    throw std::runtime_error("UFL mesh and CoordinateElement do not match.");
  }

  auto element = std::make_shared<FiniteElement>(*ufcx_element);
  assert(element);
  ufcx_dofmap* ufcx_map = space->dofmap;
  assert(ufcx_map);
  ElementDofLayout layout
      = create_element_dof_layout(*ufcx_map, mesh->topology().cell_type());
  return FunctionSpace(
      mesh, element,
      std::make_shared<DofMap>(create_dofmap(
          mesh->comm(), layout, mesh->topology(), reorder_fn, *element)));
}

/// @private
namespace impl
{
/// @private
template <typename T, typename U>
std::span<const std::uint32_t> get_cell_orientation_info(
    const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients)
{
  bool needs_dof_transformations = false;
  for (auto coeff : coefficients)
  {
    std::shared_ptr<const FiniteElement> element
        = coeff->function_space()->element();
    if (element->needs_dof_transformations())
    {
      needs_dof_transformations = true;
      break;
    }
  }

  std::span<const std::uint32_t> cell_info;
  if (needs_dof_transformations)
  {
    auto mesh = coefficients.front()->function_space()->mesh();
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(mesh->topology().get_cell_permutation_info());
  }

  return cell_info;
}

/// Pack a single coefficient for a single cell
template <typename T, int _bs>
void pack(std::span<T> coeffs, std::int32_t cell, int bs, std::span<const T> v,
          std::span<const std::uint32_t> cell_info, const DofMap& dofmap,
          auto transform)
{
  auto dofs = dofmap.cell_dofs(cell);
  for (std::size_t i = 0; i < dofs.size(); ++i)
  {
    if constexpr (_bs < 0)
    {
      const int pos_c = bs * i;
      const int pos_v = bs * dofs[i];
      for (int k = 0; k < bs; ++k)
        coeffs[pos_c + k] = v[pos_v + k];
    }
    else
    {
      const int pos_c = _bs * i;
      const int pos_v = _bs * dofs[i];
      for (int k = 0; k < _bs; ++k)
        coeffs[pos_c + k] = v[pos_v + k];
    }
  }

  transform(coeffs, cell_info, cell, 1);
}

/// @private
/// @brief  Concepts for function that returns cell index
template <typename F>
concept FetchCells = requires(F&& f, std::span<const std::int32_t> v) {
  requires std::invocable<F, std::span<const std::int32_t>>;
  {
    f(v)
  } -> std::convertible_to<std::int32_t>;
};

/// @brief Pack a single coefficient for a set of active entities.
///
/// @param[out] c The coefficient to be packed
/// @param[in] cstride The total number of coefficient values to pack
/// for each entity
/// @param[in] u The function to extract coefficient data from
/// @param[in] cell_info Array of bytes describing which transformation
/// has to be applied on the cell to map it to the reference element
/// @param[in] entities The set of active entities
/// @param[in] estride The stride for each entity in active entities.
/// @param[in] fetch_cells Function that fetches the cell index for an
/// entity in active_entities.
/// @param[in] offset The offset for c
template <typename T, typename U>
void pack_coefficient_entity(std::span<T> c, int cstride,
                             const Function<T, U>& u,
                             std::span<const std::uint32_t> cell_info,
                             std::span<const std::int32_t> entities,
                             std::size_t estride, FetchCells auto&& fetch_cells,
                             std::int32_t offset)
{
  // Read data from coefficient Function u
  std::span<const T> v = u.x()->array();
  const DofMap& dofmap = *u.function_space()->dofmap();
  std::shared_ptr<const FiniteElement> element = u.function_space()->element();
  assert(element);
  int space_dim = element->space_dimension();
  const auto transformation
      = element->get_dof_transformation_function<T>(false, true);

  const int bs = dofmap.bs();
  switch (bs)
  {
  case 1:
    for (std::size_t e = 0; e < entities.size(); e += estride)
    {
      auto entity = entities.subspan(e, estride);
      std::int32_t cell = fetch_cells(entity);
      auto cell_coeff = c.subspan((e / estride) * cstride + offset, space_dim);
      pack<T, 1>(cell_coeff, cell, bs, v, cell_info, dofmap, transformation);
    }
    break;
  case 2:
    for (std::size_t e = 0; e < entities.size(); e += estride)
    {
      auto entity = entities.subspan(e, estride);
      std::int32_t cell = fetch_cells(entity);
      auto cell_coeff = c.subspan((e / estride) * cstride + offset, space_dim);
      pack<T, 2>(cell_coeff, cell, bs, v, cell_info, dofmap, transformation);
    }
    break;
  case 3:
    for (std::size_t e = 0; e < entities.size(); e += estride)
    {
      auto entity = entities.subspan(e, estride);
      std::int32_t cell = fetch_cells(entity);
      auto cell_coeff = c.subspan(e / estride * cstride + offset, space_dim);
      pack<T, 3>(cell_coeff, cell, bs, v, cell_info, dofmap, transformation);
    }
    break;
  default:
    for (std::size_t e = 0; e < entities.size(); e += estride)
    {
      auto entity = entities.subspan(e, estride);
      std::int32_t cell = fetch_cells(entity);
      auto cell_coeff = c.subspan((e / estride) * cstride + offset, space_dim);
      pack<T, -1>(cell_coeff, cell, bs, v, cell_info, dofmap, transformation);
    }
    break;
  }
}

} // namespace impl

/// @brief Allocate storage for coefficients of a pair (integral_type,
/// id) from a fem::Form form
/// @param[in] form The Form
/// @param[in] integral_type Type of integral
/// @param[in] id The id of the integration domain
/// @return A storage container and the column stride
template <typename T, typename U>
std::pair<std::vector<T>, int>
allocate_coefficient_storage(const Form<T, U>& form, IntegralType integral_type,
                             int id)
{
  // Get form coefficient offsets and dofmaps
  const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients
      = form.coefficients();
  const std::vector<int> offsets = form.coefficient_offsets();

  std::size_t num_entities = 0;
  int cstride = 0;
  if (!coefficients.empty())
  {
    cstride = offsets.back();
    switch (integral_type)
    {
    case IntegralType::cell:
      num_entities = form.cell_domains(id).size();
      break;
    case IntegralType::exterior_facet:
      num_entities = form.exterior_facet_domains(id).size() / 2;
      break;
    case IntegralType::interior_facet:
      num_entities = form.interior_facet_domains(id).size() / 2;
      break;
    default:
      throw std::runtime_error("Could not allocate coefficient data. "
                               "Integral type not supported.");
    }
  }

  return {std::vector<T>(num_entities * cstride), cstride};
}

/// @brief Allocate memory for packed coefficients of a Form
/// @param[in] form The Form
/// @return A map from a form (integral_type, domain_id) pair to a
/// (coeffs, cstride) pair
template <typename T, typename U>
std::map<std::pair<IntegralType, int>, std::pair<std::vector<T>, int>>
allocate_coefficient_storage(const Form<T, U>& form)
{
  std::map<std::pair<IntegralType, int>, std::pair<std::vector<T>, int>> coeffs;
  for (auto integral_type : form.integral_types())
  {
    for (int id : form.integral_ids(integral_type))
    {
      coeffs.emplace_hint(
          coeffs.end(), std::pair(integral_type, id),
          allocate_coefficient_storage(form, integral_type, id));
    }
  }

  return coeffs;
}

/// @brief Pack coefficients of a Form for a given integral type and
/// domain id
/// @param[in] form The Form
/// @param[in] integral_type Type of integral
/// @param[in] id The id of the integration domain
/// @param[in] c The coefficient array
/// @param[in] cstride The coefficient stride
template <typename T, typename U>
void pack_coefficients(const Form<T, U>& form, IntegralType integral_type,
                       int id, std::span<T> c, int cstride)
{
  // Get form coefficient offsets and dofmaps
  const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients
      = form.coefficients();
  const std::vector<int> offsets = form.coefficient_offsets();

  if (!coefficients.empty())
  {
    std::span<const std::uint32_t> cell_info
        = impl::get_cell_orientation_info(coefficients);

    switch (integral_type)
    {
    case IntegralType::cell:
    {
      auto fetch_cell = [](auto entity) { return entity.front(); };
      const std::vector<std::int32_t>& cells = form.cell_domains(id);

      // Iterate over coefficients
      for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
      {
        impl::pack_coefficient_entity(c, cstride, *coefficients[coeff],
                                      cell_info, cells, 1, fetch_cell,
                                      offsets[coeff]);
      }
      break;
    }
    case IntegralType::exterior_facet:
    {
      const std::vector<std::int32_t>& facets = form.exterior_facet_domains(id);

      // Function to fetch cell index from exterior facet entity
      auto fetch_cell = [](auto entity) { return entity.front(); };

      // Iterate over coefficients
      for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
      {
        impl::pack_coefficient_entity(c, cstride, *coefficients[coeff],
                                      cell_info, facets, 2, fetch_cell,
                                      offsets[coeff]);
      }
      break;
    }
    case IntegralType::interior_facet:
    {
      const std::vector<std::int32_t>& facets = form.interior_facet_domains(id);

      // Functions to fetch cell indices from interior facet entity
      auto fetch_cell0 = [](auto entity) { return entity[0]; };
      auto fetch_cell1 = [](auto entity) { return entity[2]; };

      // Iterate over coefficients
      for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
      {
        // Pack coefficient ['+']
        impl::pack_coefficient_entity(c, 2 * cstride, *coefficients[coeff],
                                      cell_info, facets, 4, fetch_cell0,
                                      2 * offsets[coeff]);
        // Pack coefficient ['-']
        impl::pack_coefficient_entity(c, 2 * cstride, *coefficients[coeff],
                                      cell_info, facets, 4, fetch_cell1,
                                      offsets[coeff] + offsets[coeff + 1]);
      }
      break;
    }
    default:
      throw std::runtime_error(
          "Could not pack coefficient. Integral type not supported.");
    }
  }
}

/// @brief Create Expression from UFC
template <typename T, typename U = dolfinx::scalar_value_type_t<T>>
Expression<T, U> create_expression(
    const ufcx_expression& expression,
    const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients,
    const std::vector<std::shared_ptr<const Constant<T>>>& constants,
    std::shared_ptr<const mesh::Mesh<U>> mesh = nullptr,
    std::shared_ptr<const FunctionSpace<U>> argument_function_space = nullptr)
{
  if (expression.rank > 0 and !argument_function_space)
  {
    throw std::runtime_error("Expression has Argument but no Argument "
                             "function space was provided.");
  }

  const std::size_t size
      = expression.num_points * expression.topological_dimension;
  std::span<const double> X(expression.points, size);
  std::array<std::size_t, 2> Xshape
      = {static_cast<std::size_t>(expression.num_points),
         static_cast<std::size_t>(expression.topological_dimension)};

  std::vector<int> value_shape;
  for (int i = 0; i < expression.num_components; ++i)
    value_shape.push_back(expression.value_shape[i]);

  std::function<void(T*, const T*, const T*,
                     const typename scalar_value_type<T>::value_type*,
                     const int*, const std::uint8_t*)>
      tabulate_tensor = nullptr;
  if constexpr (std::is_same_v<T, float>)
    tabulate_tensor = expression.tabulate_tensor_float32;
  else if constexpr (std::is_same_v<T, std::complex<float>>)
  {
    tabulate_tensor = reinterpret_cast<void (*)(
        T*, const T*, const T*,
        const typename scalar_value_type<T>::value_type*, const int*,
        const unsigned char*)>(expression.tabulate_tensor_complex64);
  }
  else if constexpr (std::is_same_v<T, double>)
    tabulate_tensor = expression.tabulate_tensor_float64;
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    tabulate_tensor = reinterpret_cast<void (*)(
        T*, const T*, const T*,
        const typename scalar_value_type<T>::value_type*, const int*,
        const unsigned char*)>(expression.tabulate_tensor_complex128);
  }
  else
    throw std::runtime_error("Type not supported.");

  assert(tabulate_tensor);
  return Expression(coefficients, constants, X, Xshape, tabulate_tensor,
                    value_shape, mesh, argument_function_space);
}

/// @brief Create Expression from UFC input (with named coefficients and
/// constants).
template <typename T, typename U = dolfinx::scalar_value_type_t<T>>
Expression<T, U> create_expression(
    const ufcx_expression& expression,
    const std::map<std::string, std::shared_ptr<const Function<T, U>>>&
        coefficients,
    const std::map<std::string, std::shared_ptr<const Constant<T>>>& constants,
    std::shared_ptr<const mesh::Mesh<U>> mesh = nullptr,
    std::shared_ptr<const FunctionSpace<U>> argument_function_space = nullptr)
{
  // Place coefficients in appropriate order
  std::vector<std::shared_ptr<const Function<T, U>>> coeff_map;
  std::vector<std::string> coefficient_names;
  for (int i = 0; i < expression.num_coefficients; ++i)
    coefficient_names.push_back(expression.coefficient_names[i]);

  for (const std::string& name : coefficient_names)
  {
    if (auto it = coefficients.find(name); it != coefficients.end())
      coeff_map.push_back(it->second);
    else
    {
      throw std::runtime_error("Expression coefficient \"" + name
                               + "\" not provided.");
    }
  }

  // Place constants in appropriate order
  std::vector<std::shared_ptr<const Constant<T>>> const_map;
  std::vector<std::string> constant_names;
  for (int i = 0; i < expression.num_constants; ++i)
    constant_names.push_back(expression.constant_names[i]);

  for (const std::string& name : constant_names)
  {
    if (auto it = constants.find(name); it != constants.end())
      const_map.push_back(it->second);
    else
    {
      throw std::runtime_error("Expression constant \"" + name
                               + "\" not provided.");
    }
  }

  return create_expression(expression, coeff_map, const_map, mesh,
                           argument_function_space);
}

/// @warning This is subject to change
/// @brief Pack coefficients of a Form
/// @param[in] form The Form
/// @param[in] coeffs A map from a (integral_type, domain_id) pair to a
/// (coeffs, cstride) pair
template <typename T, typename U>
void pack_coefficients(const Form<T, U>& form,
                       std::map<std::pair<IntegralType, int>,
                                std::pair<std::vector<T>, int>>& coeffs)
{
  for (auto& [key, val] : coeffs)
    pack_coefficients<T>(form, key.first, key.second, val.first, val.second);
}

/// @brief Pack coefficients of a Expression u for a give list of active
/// cells
///
/// @param[in] u The Expression
/// @param[in] cells A list of active cells
/// @return A pair of the form (coeffs, cstride)
template <typename T, typename U>
std::pair<std::vector<T>, int>
pack_coefficients(const Expression<T, U>& u,
                  std::span<const std::int32_t> cells)
{
  // Get form coefficient offsets and dofmaps
  const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients
      = u.coefficients();
  const std::vector<int> offsets = u.coefficient_offsets();

  // Copy data into coefficient array
  const int cstride = offsets.back();
  std::vector<T> c(cells.size() * offsets.back());
  if (!coefficients.empty())
  {
    std::span<const std::uint32_t> cell_info
        = impl::get_cell_orientation_info(coefficients);

    // Iterate over coefficients
    for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
    {
      impl::pack_coefficient_entity(
          std::span(c), cstride, *coefficients[coeff], cell_info, cells, 1,
          [](auto entity) { return entity[0]; }, offsets[coeff]);
    }
  }
  return {std::move(c), cstride};
}

/// @brief Pack constants of u of generic type U ready for assembly
/// @warning This function is subject to change
template <typename U>
std::vector<typename U::scalar_type> pack_constants(const U& u)
{
  using T = typename U::scalar_type;
  const std::vector<std::shared_ptr<const Constant<T>>>& constants
      = u.constants();

  // Calculate size of array needed to store packed constants
  std::int32_t size = std::accumulate(constants.cbegin(), constants.cend(), 0,
                                      [](std::int32_t sum, auto& constant)
                                      { return sum + constant->value.size(); });

  // Pack constants
  std::vector<T> constant_values(size);
  std::int32_t offset = 0;
  for (auto& constant : constants)
  {
    const std::vector<T>& value = constant->value;
    std::copy(value.begin(), value.end(),
              std::next(constant_values.begin(), offset));
    offset += value.size();
  }

  return constant_values;
}

} // namespace dolfinx::fem
