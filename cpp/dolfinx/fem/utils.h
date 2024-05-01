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
#include "FunctionSpace.h"
#include "sparsitybuild.h"
#include <array>
#include <concepts>
#include <dolfinx/common/types.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Topology.h>
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
template <std::floating_point T>
class FiniteElement;
}

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::fem
{

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
get_cell_facet_pairs(std::int32_t f, std::span<const std::int32_t> cells,
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

/// @brief Given an integral type and mesh tag data, compute the
/// entities that should be integrated over.
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
/// @return List of `(integral id, entities)` pairs
/// @pre The topological dimension of the integral entity type and the
/// topological dimension of mesh tag data must be equal.
/// @pre For facet integrals, the topology facet-to-cell and
/// cell-to-facet connectivity must be computed before calling this
/// function.
std::vector<std::pair<int, std::vector<std::int32_t>>>
compute_integration_domains(IntegralType integral_type,
                            const mesh::Topology& topology,
                            std::span<const std::int32_t> entities, int dim,
                            std::span<const int> values);

/// @brief Extract test (0) and trial (1) function spaces pairs for each
/// bilinear form for a rectangular array of forms.
///
/// @param[in] a A rectangular block on bilinear forms
/// @return Rectangular array of the same shape as `a` with a pair of
/// function spaces in each array entry. If a form is null, then the
/// returned function space pair is (null, null).
template <dolfinx::scalar T, std::floating_point U>
std::vector<std::vector<std::array<std::shared_ptr<const FunctionSpace<U>>, 2>>>
extract_function_spaces(const std::vector<std::vector<const Form<T, U>*>>& a)
{
  std::vector<
      std::vector<std::array<std::shared_ptr<const FunctionSpace<U>>, 2>>>
      spaces(
          a.size(),
          std::vector<std::array<std::shared_ptr<const FunctionSpace<U>>, 2>>(
              a.front().size()));
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
/// @param[in] a A bilinear form
/// @return The corresponding sparsity pattern
template <dolfinx::scalar T, std::floating_point U>
la::SparsityPattern create_sparsity_pattern(const Form<T, U>& a)
{
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear.");
  }

  // Get dof maps and mesh
  std::array<std::reference_wrapper<const DofMap>, 2> dofmaps{
      *a.function_spaces().at(0)->dofmap(),
      *a.function_spaces().at(1)->dofmap()};
  std::shared_ptr mesh = a.mesh();
  assert(mesh);

  std::shared_ptr mesh0 = a.function_spaces().at(0)->mesh();
  assert(mesh0);
  std::shared_ptr mesh1 = a.function_spaces().at(1)->mesh();
  assert(mesh1);

  const std::set<IntegralType> types = a.integral_types();
  if (types.find(IntegralType::interior_facet) != types.end()
      or types.find(IntegralType::exterior_facet) != types.end())
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    int tdim = mesh->topology()->dim();
    mesh->topology_mutable()->create_entities(tdim - 1);
    mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
  }

  common::Timer t0("Build sparsity");

  // Get common::IndexMaps for each dimension
  const std::array index_maps{dofmaps[0].get().index_map,
                              dofmaps[1].get().index_map};
  const std::array bs
      = {dofmaps[0].get().index_map_bs(), dofmaps[1].get().index_map_bs()};

  auto extract_cells = [](std::span<const std::int32_t> facets)
  {
    assert(facets.size() % 2 == 0);
    std::vector<std::int32_t> cells;
    cells.reserve(facets.size() / 2);
    for (std::size_t i = 0; i < facets.size(); i += 2)
      cells.push_back(facets[i]);
    return cells;
  };

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
        sparsitybuild::cells(
            pattern, {a.domain(type, id, *mesh0), a.domain(type, id, *mesh1)},
            {{dofmaps[0], dofmaps[1]}});
      }
      break;
    case IntegralType::interior_facet:
      for (int id : ids)
      {
        sparsitybuild::interior_facets(
            pattern,
            {extract_cells(a.domain(type, id, *mesh0)),
             extract_cells(a.domain(type, id, *mesh1))},
            {{dofmaps[0], dofmaps[1]}});
      }
      break;
    case IntegralType::exterior_facet:
      for (int id : ids)
      {
        sparsitybuild::cells(pattern,
                             {extract_cells(a.domain(type, id, *mesh0)),
                              extract_cells(a.domain(type, id, *mesh1))},
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

/// Create an ElementDofLayout from a FiniteElement
template <std::floating_point T>
ElementDofLayout create_element_dof_layout(const fem::FiniteElement<T>& element,
                                           const std::vector<int>& parent_map
                                           = {})
{
  // Create subdofmaps and compute offset
  std::vector<int> offsets(1, 0);
  std::vector<dolfinx::fem::ElementDofLayout> sub_doflayout;
  int bs = element.block_size();
  for (int i = 0; i < element.num_sub_elements(); ++i)
  {
    // The ith sub-element. For mixed elements this is subelements()[i]. For
    // blocked elements, the sub-element will always be the same, so we'll use
    // sub_elements()[0]
    std::shared_ptr<const fem::FiniteElement<T>> sub_e
        = element.sub_elements()[bs > 1 ? 0 : i];

    // In a mixed element DOFs are ordered element by element, so the offset to
    // the next sub-element is sub_e->space_dimension(). Blocked elements use
    // xxyyzz ordering, so the offset to the next sub-element is 1

    std::vector<int> parent_map_sub(sub_e->space_dimension(), offsets.back());
    for (std::size_t j = 0; j < parent_map_sub.size(); ++j)
      parent_map_sub[j] += bs * j;
    offsets.push_back(offsets.back() + (bs > 1 ? 1 : sub_e->space_dimension()));
    sub_doflayout.push_back(
        dolfinx::fem::create_element_dof_layout(*sub_e, parent_map_sub));
  }

  return ElementDofLayout(bs, element.entity_dofs(),
                          element.entity_closure_dofs(), parent_map,
                          sub_doflayout);
}

/// @brief Create a dof map on mesh
/// @param[in] comm MPI communicator
/// @param[in] layout Dof layout on an element
/// @param[in] topology Mesh topology
/// @param[in] permute_inv Function to un-permute dofs. `nullptr`
/// when transformation is not required.
/// @param[in] reorder_fn Graph reordering function called on the dofmap
/// @return A new dof map
DofMap create_dofmap(
    MPI_Comm comm, const ElementDofLayout& layout, mesh::Topology& topology,
    std::function<void(std::span<std::int32_t>, std::uint32_t)> permute_inv,
    std::function<std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>
        reorder_fn);

/// @brief Create a set of dofmaps on a given topology
/// @param[in] comm MPI communicator
/// @param[in] layouts Dof layout on each element type
/// @param[in] topology Mesh topology
/// @param[in] permute_inv Function to un-permute dofs. `nullptr`
/// when transformation is not required.
/// @param[in] reorder_fn Graph reordering function called on the dofmaps
/// @return The list of new dof maps
/// @note The number of layouts must match the number of cell types in the
/// topology
std::vector<DofMap> create_dofmaps(
    MPI_Comm comm, const std::vector<ElementDofLayout>& layouts,
    mesh::Topology& topology,
    std::function<void(std::span<std::int32_t>, std::uint32_t)> permute_inv,
    std::function<std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>
        reorder_fn);

/// Get the name of each coefficient in a UFC form
/// @param[in] ufcx_form The UFC form
/// @return The name of each coefficient
std::vector<std::string> get_coefficient_names(const ufcx_form& ufcx_form);

/// @brief Get the name of each constant in a UFC form
/// @param[in] ufcx_form The UFC form
/// @return The name of each constant
std::vector<std::string> get_constant_names(const ufcx_form& ufcx_form);

/// @brief Create a Form from UFCx input with coefficients and constants
/// passed in the required order.
///
/// Use fem::create_form to create a fem::Form with coefficients and
/// constants associated with the name/string.
///
/// @param[in] ufcx_form The UFCx form.
/// @param[in] spaces Vector of function spaces. The number of spaces is
/// equal to the rank of the form.
/// @param[in] coefficients Coefficient fields in the form.
/// @param[in] constants Spatial constants in the form.
/// @param[in] subdomains Subdomain markers.
/// @param[in] entity_maps The entity maps for the form. Empty for
/// single domain problems.
/// @param[in] mesh The mesh of the domain.
///
/// @pre Each value in `subdomains` must be sorted by domain id.
template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
Form<T, U> create_form_factory(
    const ufcx_form& ufcx_form,
    const std::vector<std::shared_ptr<const FunctionSpace<U>>>& spaces,
    const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients,
    const std::vector<std::shared_ptr<const Constant<T>>>& constants,
    const std::map<
        IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>&
        subdomains,
    const std::map<std::shared_ptr<const mesh::Mesh<U>>,
                   std::span<const std::int32_t>>& entity_maps,
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
  for (std::size_t i = 0; i < spaces.size(); ++i)
  {
    assert(spaces[i]->element());
    if (auto element_hash = ufcx_form.finite_element_hashes[i];
        element_hash != 0
        and element_hash != spaces[i]->element()->basix_element().hash())
    {
      throw std::runtime_error("Cannot create form. Elements are different to "
                               "those used to compile the form.");
    }
  }

  // Extract mesh from FunctionSpace, and check they are the same
  if (!mesh and !spaces.empty())
    mesh = spaces[0]->mesh();
  for (auto& V : spaces)
  {
    if (mesh != V->mesh() and entity_maps.find(V->mesh()) == entity_maps.end())
      throw std::runtime_error(
          "Incompatible mesh. entity_maps must be provided.");
  }
  if (!mesh)
    throw std::runtime_error("No mesh could be associated with the Form.");

  auto topology = mesh->topology();
  assert(topology);
  const int tdim = topology->dim();

  const int* integral_offsets = ufcx_form.form_integral_offsets;
  std::vector<int> num_integrals_type(3);
  for (int i = 0; i < 3; ++i)
    num_integrals_type[i] = integral_offsets[i + 1] - integral_offsets[i];

  // Create facets, if required
  if (num_integrals_type[exterior_facet] > 0
      or num_integrals_type[interior_facet] > 0)
  {
    mesh->topology_mutable()->create_entities(tdim - 1);
    mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
    mesh->topology_mutable()->create_connectivity(tdim, tdim - 1);
  }

  // Get list of integral IDs, and load tabulate tensor into memory for
  // each
  using kern_t = std::function<void(T*, const T*, const T*, const U*,
                                    const int*, const std::uint8_t*)>;
  std::map<IntegralType, std::vector<integral_data<T, U>>> integrals;

  // Attach cell kernels
  bool needs_facet_permutations = false;
  std::vector<std::int32_t> default_cells;
  {
    std::span<const int> ids(ufcx_form.form_integral_ids
                                 + integral_offsets[cell],
                             num_integrals_type[cell]);
    auto itg = integrals.insert({IntegralType::cell, {}});
    auto sd = subdomains.find(IntegralType::cell);
    for (int i = 0; i < num_integrals_type[cell]; ++i)
    {
      const int id = ids[i];
      ufcx_integral* integral
          = ufcx_form.form_integrals[integral_offsets[cell] + i];
      assert(integral);

      kern_t k = nullptr;
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
      if (!k)
      {
        throw std::runtime_error(
            "UFCx kernel function is NULL. Check requested types.");
      }

      // Build list of entities to assemble over
      if (id == -1)
      {
        // Default kernel, operates on all (owned) cells
        assert(topology->index_map(tdim));
        default_cells.resize(topology->index_map(tdim)->size_local(), 0);
        std::iota(default_cells.begin(), default_cells.end(), 0);
        itg.first->second.emplace_back(id, k, default_cells);
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
  std::vector<std::int32_t> default_facets_ext;
  {
    std::span<const int> ids(ufcx_form.form_integral_ids
                                 + integral_offsets[exterior_facet],
                             num_integrals_type[exterior_facet]);
    auto itg = integrals.insert({IntegralType::exterior_facet, {}});
    auto sd = subdomains.find(IntegralType::exterior_facet);
    for (int i = 0; i < num_integrals_type[exterior_facet]; ++i)
    {
      const int id = ids[i];
      ufcx_integral* integral
          = ufcx_form.form_integrals[integral_offsets[exterior_facet] + i];
      assert(integral);

      kern_t k = nullptr;
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
      const std::vector bfacets = mesh::exterior_facet_indices(*topology);
      auto f_to_c = topology->connectivity(tdim - 1, tdim);
      assert(f_to_c);
      auto c_to_f = topology->connectivity(tdim, tdim - 1);
      assert(c_to_f);
      if (id == -1)
      {
        // Default kernel, operates on all (owned) exterior facets
        default_facets_ext.reserve(2 * bfacets.size());
        for (std::int32_t f : bfacets)
        {
          // There will only be one pair for an exterior facet integral
          auto pair
              = impl::get_cell_facet_pairs<1>(f, f_to_c->links(f), *c_to_f);
          default_facets_ext.insert(default_facets_ext.end(), pair.begin(),
                                    pair.end());
        }
        itg.first->second.emplace_back(id, k, default_facets_ext);
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
  std::vector<std::int32_t> default_facets_int;
  {
    std::span<const int> ids(ufcx_form.form_integral_ids
                                 + integral_offsets[interior_facet],
                             num_integrals_type[interior_facet]);
    auto itg = integrals.insert({IntegralType::interior_facet, {}});
    auto sd = subdomains.find(IntegralType::interior_facet);
    for (int i = 0; i < num_integrals_type[interior_facet]; ++i)
    {
      const int id = ids[i];
      ufcx_integral* integral
          = ufcx_form.form_integrals[integral_offsets[interior_facet] + i];
      assert(integral);

      kern_t k = nullptr;
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
      auto f_to_c = topology->connectivity(tdim - 1, tdim);
      assert(f_to_c);
      auto c_to_f = topology->connectivity(tdim, tdim - 1);
      assert(c_to_f);
      if (id == -1)
      {
        // Default kernel, operates on all (owned) interior facets
        assert(topology->index_map(tdim - 1));
        std::int32_t num_facets = topology->index_map(tdim - 1)->size_local();
        default_facets_int.reserve(4 * num_facets);
        for (std::int32_t f = 0; f < num_facets; ++f)
        {
          if (f_to_c->num_links(f) == 2)
          {
            auto pairs
                = impl::get_cell_facet_pairs<2>(f, f_to_c->links(f), *c_to_f);
            default_facets_int.insert(default_facets_int.end(), pairs.begin(),
                                      pairs.end());
          }
        }
        itg.first->second.emplace_back(id, k, default_facets_int);
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

  return Form<T, U>(spaces, integrals, coefficients, constants,
                    needs_facet_permutations, entity_maps, mesh);
}

/// @brief Create a Form from UFC input with coefficients and constants
/// resolved by name.
/// @param[in] ufcx_form UFC form
/// @param[in] spaces Function spaces for the Form arguments.
/// @param[in] coefficients Coefficient fields in the form (by name).
/// @param[in] constants Spatial constants in the form (by name).
/// @param[in] subdomains Subdomain markers.
/// @pre Each value in `subdomains` must be sorted by domain id.
/// @param[in] mesh Mesh of the domain. This is required if the form has
/// no arguments, e.g. a functional.
/// @return A Form
template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
Form<T, U> create_form(
    const ufcx_form& ufcx_form,
    const std::vector<std::shared_ptr<const FunctionSpace<U>>>& spaces,
    const std::map<std::string, std::shared_ptr<const Function<T, U>>>&
        coefficients,
    const std::map<std::string, std::shared_ptr<const Constant<T>>>& constants,
    const std::map<
        IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>&
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

  return create_form_factory(ufcx_form, spaces, coeff_map, const_map,
                             subdomains, {}, mesh);
}

/// @brief Create a Form using a factory function that returns a pointer
/// to a `ufcx_form`.
///
/// Coefficients and constants are resolved by name/string.
///
/// @param[in] fptr Pointer to a function returning a pointer to
/// ufcx_form.
/// @param[in] spaces Function spaces for the Form arguments.
/// @param[in] coefficients Coefficient fields in the form (by name),
/// @param[in] constants Spatial constants in the form (by name),
/// @param[in] subdomains Subdomain markers.
/// @pre Each value in `subdomains` must be sorted by domain id.
/// @param[in] mesh Mesh of the domain. This is required if the form has
/// no arguments, e.g. a functional.
/// @return A Form
template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
Form<T, U> create_form(
    ufcx_form* (*fptr)(),
    const std::vector<std::shared_ptr<const FunctionSpace<U>>>& spaces,
    const std::map<std::string, std::shared_ptr<const Function<T, U>>>&
        coefficients,
    const std::map<std::string, std::shared_ptr<const Constant<T>>>& constants,
    const std::map<
        IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>&
        subdomains,
    std::shared_ptr<const mesh::Mesh<U>> mesh = nullptr)
{
  ufcx_form* form = fptr();
  Form<T, U> L = create_form<T, U>(*form, spaces, coefficients, constants,
                                   subdomains, mesh);
  std::free(form);
  return L;
}

/// @brief Create a function space from a Basix element.
/// @param[in] mesh Mesh
/// @param[in] e Basix finite element.
/// @param[in] value_shape Value shape for 'blocked' elements, e.g.
/// vector-valued Lagrange elements where each component for the vector
/// field is a Lagrange element. For example, a vector-valued element in
/// 3D will have `value_shape` equal to `{3}`, and for a second-order
/// tensor element in 2D `value_shape` equal to `{2, 2}`.
/// @param[in] reorder_fn The graph reordering function to call on the
/// dofmap. If `nullptr`, the default re-ordering is used.
/// @return The created function space
template <std::floating_point T>
FunctionSpace<T> create_functionspace(
    std::shared_ptr<mesh::Mesh<T>> mesh, const basix::FiniteElement<T>& e,
    const std::vector<std::size_t>& value_shape = {},
    std::function<std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>
        reorder_fn
    = nullptr)
{
  if (!e.value_shape().empty() and !value_shape.empty())
  {
    throw std::runtime_error(
        "Cannot specify value shape for non-scalar base element.");
  }

  std::size_t bs = value_shape.empty()
                       ? 1
                       : std::accumulate(value_shape.begin(), value_shape.end(),
                                         1, std::multiplies{});

  // Create a DOLFINx element
  auto _e = std::make_shared<const FiniteElement<T>>(e, bs);
  assert(_e);

  const std::vector<std::size_t> _value_shape
      = (value_shape.empty() and !e.value_shape().empty())
            ? fem::compute_value_shape(_e, mesh->topology()->dim(),
                                       mesh->geometry().dim())
            : value_shape;

  // Create UFC subdofmaps and compute offset
  const int num_sub_elements = _e->num_sub_elements();
  std::vector<ElementDofLayout> sub_doflayout;
  sub_doflayout.reserve(num_sub_elements);
  for (int i = 0; i < num_sub_elements; ++i)
  {
    auto sub_element = _e->extract_sub_element({i});
    std::vector<int> parent_map_sub(sub_element->space_dimension());
    for (std::size_t j = 0; j < parent_map_sub.size(); ++j)
      parent_map_sub[j] = i + _e->block_size() * j;
    sub_doflayout.emplace_back(1, e.entity_dofs(), e.entity_closure_dofs(),
                               parent_map_sub, std::vector<ElementDofLayout>());
  }

  // Create a dofmap
  ElementDofLayout layout(_e->block_size(), e.entity_dofs(),
                          e.entity_closure_dofs(), {}, sub_doflayout);
  std::function<void(std::span<std::int32_t>, std::uint32_t)> permute_inv
      = nullptr;
  if (_e->needs_dof_permutations())
    permute_inv = _e->dof_permutation_fn(true, true);
  assert(mesh);
  assert(mesh->topology());
  auto dofmap = std::make_shared<const DofMap>(create_dofmap(
      mesh->comm(), layout, *mesh->topology(), permute_inv, reorder_fn));
  return FunctionSpace(mesh, _e, dofmap, _value_shape);
}

/// @private
namespace impl
{
/// @private
template <dolfinx::scalar T, std::floating_point U>
std::span<const std::uint32_t>
get_cell_orientation_info(const Function<T, U>& coefficient)
{
  std::span<const std::uint32_t> cell_info;
  auto element = coefficient.function_space()->element();
  if (element->needs_dof_transformations())
  {
    auto mesh = coefficient.function_space()->mesh();
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  return cell_info;
}

/// Pack a single coefficient for a single cell
template <dolfinx::scalar T, int _bs>
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
  { f(v) } -> std::convertible_to<std::int32_t>;
};

/// @brief Pack a single coefficient for a set of active entities.
///
/// @param[out] c Coefficient to be packed.
/// @param[in] cstride Total number of coefficient values to pack for
/// each entity.
/// @param[in] u Function to extract coefficient data from.
/// @param[in] cell_info Array of bytes describing which transformation
/// has to be applied on the cell to map it to the reference element.
/// @param[in] entities Set of active entities.
/// @param[in] estride Stride for each entity in active entities.
/// @param[in] fetch_cells Function that fetches the cell index for an
/// entity in active_entities.
/// @param[in] offset The offset for c.
template <dolfinx::scalar T, std::floating_point U>
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
  auto element = u.function_space()->element();
  assert(element);
  int space_dim = element->space_dimension();

  // Transformation from conforming degrees-of-freedom to reference
  // degrees-of-freedom
  auto transformation
      = element->template dof_transformation_fn<T>(doftransform::transpose);
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

/// @brief Allocate storage for coefficients of a pair `(integral_type,
/// id)` from a Form.
/// @param[in] form The Form
/// @param[in] integral_type Type of integral
/// @param[in] id The id of the integration domain
/// @return A storage container and the column stride
template <dolfinx::scalar T, std::floating_point U>
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
    num_entities = form.domain(integral_type, id).size();
    if (integral_type == IntegralType::exterior_facet
        or integral_type == IntegralType::interior_facet)
    {
      num_entities /= 2;
    }
  }

  return {std::vector<T>(num_entities * cstride), cstride};
}

/// @brief Allocate memory for packed coefficients of a Form.
/// @param[in] form The Form
/// @return Map from a form `(integral_type, domain_id)` pair to a
/// `(coeffs, cstride)` pair
template <dolfinx::scalar T, std::floating_point U>
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
template <dolfinx::scalar T, std::floating_point U>
void pack_coefficients(const Form<T, U>& form, IntegralType integral_type,
                       int id, std::span<T> c, int cstride)
{
  // Get form coefficient offsets and dofmaps
  const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients
      = form.coefficients();
  const std::vector<int> offsets = form.coefficient_offsets();

  if (!coefficients.empty())
  {
    switch (integral_type)
    {
    case IntegralType::cell:
    {
      // Iterate over coefficients
      for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
      {
        auto mesh = coefficients[coeff]->function_space()->mesh();
        assert(mesh);
        std::vector<std::int32_t> cells
            = form.domain(IntegralType::cell, id, *mesh);
        std::span<const std::uint32_t> cell_info
            = impl::get_cell_orientation_info(*coefficients[coeff]);
        impl::pack_coefficient_entity(
            c, cstride, *coefficients[coeff], cell_info, cells, 1,
            [](auto entity) { return entity.front(); }, offsets[coeff]);
      }
      break;
    }
    case IntegralType::exterior_facet:
    {
      // Iterate over coefficients
      for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
      {
        auto mesh = coefficients[coeff]->function_space()->mesh();
        std::vector<std::int32_t> facets
            = form.domain(IntegralType::exterior_facet, id, *mesh);
        std::span<const std::uint32_t> cell_info
            = impl::get_cell_orientation_info(*coefficients[coeff]);
        impl::pack_coefficient_entity(
            c, cstride, *coefficients[coeff], cell_info, facets, 2,
            [](auto entity) { return entity.front(); }, offsets[coeff]);
      }
      break;
    }
    case IntegralType::interior_facet:
    {
      // Iterate over coefficients
      for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
      {
        auto mesh = coefficients[coeff]->function_space()->mesh();
        std::vector<std::int32_t> facets
            = form.domain(IntegralType::interior_facet, id, *mesh);
        std::span<const std::uint32_t> cell_info
            = impl::get_cell_orientation_info(*coefficients[coeff]);

        // Pack coefficient ['+']
        impl::pack_coefficient_entity(
            c, 2 * cstride, *coefficients[coeff], cell_info, facets, 4,
            [](auto entity) { return entity[0]; }, 2 * offsets[coeff]);
        // Pack coefficient ['-']
        impl::pack_coefficient_entity(
            c, 2 * cstride, *coefficients[coeff], cell_info, facets, 4,
            [](auto entity) { return entity[2]; },
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
template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
Expression<T, U> create_expression(
    const ufcx_expression& e,
    const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients,
    const std::vector<std::shared_ptr<const Constant<T>>>& constants,
    std::shared_ptr<const FunctionSpace<U>> argument_function_space = nullptr)
{
  if (e.rank > 0 and !argument_function_space)
  {
    throw std::runtime_error("Expression has Argument but no Argument "
                             "function space was provided.");
  }

  std::vector<U> X(e.points, e.points + e.num_points * e.entity_dimension);
  std::array<std::size_t, 2> Xshape
      = {static_cast<std::size_t>(e.num_points),
         static_cast<std::size_t>(e.entity_dimension)};
  std::vector<int> value_shape(e.value_shape, e.value_shape + e.num_components);
  std::function<void(T*, const T*, const T*,
                     const typename scalar_value_type<T>::value_type*,
                     const int*, const std::uint8_t*)>
      tabulate_tensor = nullptr;
  if constexpr (std::is_same_v<T, float>)
    tabulate_tensor = e.tabulate_tensor_float32;
  else if constexpr (std::is_same_v<T, std::complex<float>>)
  {
    tabulate_tensor = reinterpret_cast<void (*)(
        T*, const T*, const T*,
        const typename scalar_value_type<T>::value_type*, const int*,
        const unsigned char*)>(e.tabulate_tensor_complex64);
  }
  else if constexpr (std::is_same_v<T, double>)
    tabulate_tensor = e.tabulate_tensor_float64;
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    tabulate_tensor = reinterpret_cast<void (*)(
        T*, const T*, const T*,
        const typename scalar_value_type<T>::value_type*, const int*,
        const unsigned char*)>(e.tabulate_tensor_complex128);
  }
  else
    throw std::runtime_error("Type not supported.");

  assert(tabulate_tensor);
  return Expression(coefficients, constants, std::span<const U>(X), Xshape,
                    tabulate_tensor, value_shape, argument_function_space);
}

/// @brief Create Expression from UFC input (with named coefficients and
/// constants).
template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
Expression<T, U> create_expression(
    const ufcx_expression& e,
    const std::map<std::string, std::shared_ptr<const Function<T, U>>>&
        coefficients,
    const std::map<std::string, std::shared_ptr<const Constant<T>>>& constants,
    std::shared_ptr<const FunctionSpace<U>> argument_function_space = nullptr)
{
  // Place coefficients in appropriate order
  std::vector<std::shared_ptr<const Function<T, U>>> coeff_map;
  std::vector<std::string> coefficient_names;
  for (int i = 0; i < e.num_coefficients; ++i)
    coefficient_names.push_back(e.coefficient_names[i]);

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
  for (int i = 0; i < e.num_constants; ++i)
    constant_names.push_back(e.constant_names[i]);

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

  return create_expression(e, coeff_map, const_map, argument_function_space);
}

/// @warning This is subject to change
/// @brief Pack coefficients of a Form
/// @param[in] form The Form
/// @param[in,out] coeffs A map from an (integral_type, domain_id) pair to a
/// (coeffs, cstride) pair. `coeffs` is a storage container representing
/// an array of shape (num_int_entities, cstride) in which to pack the
/// coefficient data, where num_int_entities is the number of entities
/// being integrated over and cstride is the number of coefficient data
/// entries per integration entity. `coeffs` is flattened into row-major
/// layout.
template <dolfinx::scalar T, std::floating_point U>
void pack_coefficients(const Form<T, U>& form,
                       std::map<std::pair<IntegralType, int>,
                                std::pair<std::vector<T>, int>>& coeffs)
{
  for (auto& [key, val] : coeffs)
    pack_coefficients<T>(form, key.first, key.second, val.first, val.second);
}

/// @brief Pack coefficients of a Expression u for a give list of active
/// entities.
///
/// @param[in] e The Expression
/// @param[in] entities A list of active entities
/// @param[in] estride Stride for each entity in active entities (1 for cells, 2
/// for facets)
/// @return A pair of the form (coeffs, cstride)
template <dolfinx::scalar T, std::floating_point U>
std::pair<std::vector<T>, int>
pack_coefficients(const Expression<T, U>& e,
                  std::span<const std::int32_t> entities, std::size_t estride)
{
  // Get form coefficient offsets and dofmaps
  const std::vector<std::shared_ptr<const Function<T, U>>>& coeffs
      = e.coefficients();
  const std::vector<int> offsets = e.coefficient_offsets();

  // Copy data into coefficient array
  const int cstride = offsets.back();
  std::vector<T> c(entities.size() / estride * offsets.back());
  if (!coeffs.empty())
  {
    // Iterate over coefficients
    for (std::size_t coeff = 0; coeff < coeffs.size(); ++coeff)
    {
      std::span<const std::uint32_t> cell_info
          = impl::get_cell_orientation_info(*coeffs[coeff]);

      impl::pack_coefficient_entity(
          std::span(c), cstride, *coeffs[coeff], cell_info, entities, estride,
          [](auto entity) { return entity[0]; }, offsets[coeff]);
    }
  }
  return {std::move(c), cstride};
}

/// @brief Pack constants of u into a single array ready for assembly.
/// @warning This function is subject to change.
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
