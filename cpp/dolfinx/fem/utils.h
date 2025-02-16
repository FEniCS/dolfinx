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
#include <algorithm>
#include <array>
#include <concepts>
#include <dolfinx/common/types.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
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
template <dolfinx::scalar T, std::floating_point U>
class Expression;

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

/// @brief Given an integral type and a set of entities, computes and
/// return data for the entities that should be integrated over.
///
/// This function returns a list data, for each entity in  `entities`,
/// that is used in assembly. For cell integrals it is simply the cell
/// cell indices. For exterior facet integrals, a list of `(cell_index,
/// local_facet_index)` pairs is returned. For interior facet integrals,
/// a list of `(cell_index0, local_facet_index0, cell_index1,
/// local_facet_index1)` tuples is returned.
/// The data computed by this function is typically used as input to
/// fem::create_form.
///
/// @note Owned mesh entities only are returned. Ghost entities are not
/// included.
///
/// @pre For facet integrals, the topology facet-to-cell and
/// cell-to-facet connectivity must be computed before calling this
/// function.
///
/// @param[in] integral_type Integral type.
/// @param[in] topology Mesh topology.
/// @param[in] entities List of mesh entities. For
/// `integral_type==IntegralType::cell`, `entities` should be cell
/// indices. For other `IntegralType`, `entities` should be facet
/// indices.
/// @return List of integration entity data.
std::vector<std::int32_t>
compute_integration_domains(IntegralType integral_type,
                            const mesh::Topology& topology,
                            std::span<const std::int32_t> entities);

/// @brief Extract test (0) and trial (1) function spaces pairs for each
/// bilinear form for a rectangular array of forms.
///
/// @param[in] a A rectangular block on bilinear forms.
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
  std::shared_ptr mesh = a.mesh();
  assert(mesh);

  // Get index maps and block sizes from the DOF maps. Note that in
  // mixed-topology meshes, despite there being multiple DOF maps, the
  // index maps and block sizes are the same.
  std::array<std::reference_wrapper<const DofMap>, 2> dofmaps{
      *a.function_spaces().at(0)->dofmaps(0),
      *a.function_spaces().at(1)->dofmaps(0)};

  const std::array index_maps{dofmaps[0].get().index_map,
                              dofmaps[1].get().index_map};
  const std::array bs
      = {dofmaps[0].get().index_map_bs(), dofmaps[1].get().index_map_bs()};

  la::SparsityPattern pattern(mesh->comm(), index_maps, bs);
  build_sparsity_pattern(pattern, a);
  return pattern;
}

/// @brief Build a sparsity pattern for a given form.
/// @note The pattern is not finalised, i.e. the caller is responsible
/// for calling SparsityPattern::assemble.
/// @param[in] pattern The sparsity pattern to add to
/// @param[in] a A bilinear form
template <dolfinx::scalar T, std::floating_point U>
void build_sparsity_pattern(la::SparsityPattern& pattern, const Form<T, U>& a)
{
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear.");
  }

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

  auto extract_cells = [](std::span<const std::int32_t> facets)
  {
    assert(facets.size() % 2 == 0);
    std::vector<std::int32_t> cells;
    cells.reserve(facets.size() / 2);
    for (std::size_t i = 0; i < facets.size(); i += 2)
      cells.push_back(facets[i]);
    return cells;
  };

  const int num_cell_types = mesh->topology()->cell_types().size();
  for (int cell_type_idx = 0; cell_type_idx < num_cell_types; ++cell_type_idx)
  {
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps{
        *a.function_spaces().at(0)->dofmaps(cell_type_idx),
        *a.function_spaces().at(1)->dofmaps(cell_type_idx)};

    // Create and build sparsity pattern
    for (auto type : types)
    {
      std::vector<int> ids = a.integral_ids(type);
      switch (type)
      {
      case IntegralType::cell:
        for (int id : ids)
        {
          sparsitybuild::cells(pattern,
                               {a.domain(type, id, cell_type_idx, *mesh0),
                                a.domain(type, id, cell_type_idx, *mesh1)},
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
  }

  t0.stop();
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
/// @param[in] ufcx_forms A list of UFCx forms, one for each cell type.
/// @param[in] spaces Vector of function spaces. The number of spaces is
/// equal to the rank of the form.
/// @param[in] coefficients Coefficient fields in the form.
/// @param[in] constants Spatial constants in the form.
/// @param[in] subdomains Subdomain markers. The data can be computed
/// using fem::compute_integration_domains.
/// @param[in] entity_maps The entity maps for the form. Empty for
/// single domain problems.
/// @param[in] mesh The mesh of the domain.
///
/// @pre Each value in `subdomains` must be sorted by domain id.
template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
Form<T, U> create_form_factory(
    const std::vector<std::reference_wrapper<const ufcx_form>>& ufcx_forms,
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
  for (const ufcx_form& ufcx_form : ufcx_forms)
  {
    if (ufcx_form.rank != (int)spaces.size())
      throw std::runtime_error("Wrong number of argument spaces for Form.");
    if (ufcx_form.num_coefficients != (int)coefficients.size())
    {
      throw std::runtime_error("Mismatch between number of expected and "
                               "provided Form coefficients.");
    }
    if (ufcx_form.num_constants != (int)constants.size())
    {
      throw std::runtime_error(
          "Mismatch between number of expected and provided Form constants.");
    }
  }

  // Check argument function spaces
  for (std::size_t form_idx = 0; form_idx < ufcx_forms.size(); ++form_idx)
  {
    for (std::size_t i = 0; i < spaces.size(); ++i)
    {
      assert(spaces[i]->elements(form_idx));
      if (auto element_hash
          = ufcx_forms[form_idx].get().finite_element_hashes[i];
          element_hash != 0
          and element_hash
                  != spaces[i]->elements(form_idx)->basix_element().hash())
      {
        throw std::runtime_error(
            "Cannot create form. Elements are different to "
            "those used to compile the form.");
      }
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

  // NOTE: This assumes all forms in mixed-topology meshes have the same
  // integral offsets. Since the UFL forms for each type of cell should be
  // the same, I think this assumption is OK.
  const int* integral_offsets = ufcx_forms[0].get().form_integral_offsets;
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
  {
    std::vector<std::int32_t> default_cells;
    std::span<const int> ids(ufcx_forms[0].get().form_integral_ids
                                 + integral_offsets[cell],
                             num_integrals_type[cell]);
    auto itg = integrals.insert({IntegralType::cell, {}});
    auto sd = subdomains.find(IntegralType::cell);
    for (std::size_t form_idx = 0; form_idx < ufcx_forms.size(); ++form_idx)
    {
      const ufcx_form& ufcx_form = ufcx_forms[form_idx];
      for (int i = 0; i < num_integrals_type[cell]; ++i)
      {
        const int id = ids[i];
        ufcx_integral* integral
            = ufcx_form.form_integrals[integral_offsets[cell] + i];
        assert(integral);

        // Build list of active coefficients
        std::vector<int> active_coeffs;
        for (int j = 0; j < ufcx_form.num_coefficients; ++j)
        {
          if (integral->enabled_coefficients[j])
            active_coeffs.push_back(j);
        }

        kern_t k = nullptr;
        if constexpr (std::is_same_v<T, float>)
          k = integral->tabulate_tensor_float32;
#ifndef DOLFINX_NO_STDC_COMPLEX_KERNELS
        else if constexpr (std::is_same_v<T, std::complex<float>>)
        {
          k = reinterpret_cast<void (*)(
              T*, const T*, const T*,
              const typename scalar_value_type<T>::value_type*, const int*,
              const unsigned char*)>(integral->tabulate_tensor_complex64);
        }
#endif // DOLFINX_NO_STDC_COMPLEX_KERNELS
        else if constexpr (std::is_same_v<T, double>)
          k = integral->tabulate_tensor_float64;
#ifndef DOLFINX_NO_STDC_COMPLEX_KERNELS
        else if constexpr (std::is_same_v<T, std::complex<double>>)
        {
          k = reinterpret_cast<void (*)(
              T*, const T*, const T*,
              const typename scalar_value_type<T>::value_type*, const int*,
              const unsigned char*)>(integral->tabulate_tensor_complex128);
        }
#endif // DOLFINX_NO_STDC_COMPLEX_KERNELS

        if (!k)
        {
          throw std::runtime_error(
              "UFCx kernel function is NULL. Check requested types.");
        }

        // Build list of entities to assemble over
        if (id == -1)
        {
          // Default kernel, operates on all (owned) cells
          assert(topology->index_maps(tdim).at(form_idx));
          default_cells.resize(
              topology->index_maps(tdim).at(form_idx)->size_local(), 0);
          std::iota(default_cells.begin(), default_cells.end(), 0);
          itg.first->second.emplace_back(id, k, default_cells, active_coeffs);
        }
        else if (sd != subdomains.end())
        {
          // NOTE: This requires that pairs are sorted
          auto it = std::ranges::lower_bound(sd->second, id, std::less<>{},
                                             [](auto& a) { return a.first; });
          if (it != sd->second.end() and it->first == id)
            itg.first->second.emplace_back(id, k, it->second, active_coeffs);
        }

        if (integral->needs_facet_permutations)
          needs_facet_permutations = true;
      }
    }
  }

  // Attach exterior facet kernels
  std::vector<std::int32_t> default_facets_ext;
  {
    std::span<const int> ids(ufcx_forms[0].get().form_integral_ids
                                 + integral_offsets[exterior_facet],
                             num_integrals_type[exterior_facet]);
    auto itg = integrals.insert({IntegralType::exterior_facet, {}});
    auto sd = subdomains.find(IntegralType::exterior_facet);
    for (std::size_t form_idx = 0; form_idx < ufcx_forms.size(); ++form_idx)
    {
      const ufcx_form& ufcx_form = ufcx_forms[form_idx];
      for (int i = 0; i < num_integrals_type[exterior_facet]; ++i)
      {
        const int id = ids[i];
        ufcx_integral* integral
            = ufcx_form.form_integrals[integral_offsets[exterior_facet] + i];
        assert(integral);
        std::vector<int> active_coeffs;
        for (int j = 0; j < ufcx_form.num_coefficients; ++j)
        {
          if (integral->enabled_coefficients[j])
            active_coeffs.push_back(j);
        }

        kern_t k = nullptr;
        if constexpr (std::is_same_v<T, float>)
          k = integral->tabulate_tensor_float32;
#ifndef DOLFINX_NO_STDC_COMPLEX_KERNELS
        else if constexpr (std::is_same_v<T, std::complex<float>>)
        {
          k = reinterpret_cast<void (*)(
              T*, const T*, const T*,
              const typename scalar_value_type<T>::value_type*, const int*,
              const unsigned char*)>(integral->tabulate_tensor_complex64);
        }
#endif // DOLFINX_NO_STDC_COMPLEX_KERNELS
        else if constexpr (std::is_same_v<T, double>)
          k = integral->tabulate_tensor_float64;
#ifndef DOLFINX_NO_STDC_COMPLEX_KERNELS
        else if constexpr (std::is_same_v<T, std::complex<double>>)
        {
          k = reinterpret_cast<void (*)(
              T*, const T*, const T*,
              const typename scalar_value_type<T>::value_type*, const int*,
              const unsigned char*)>(integral->tabulate_tensor_complex128);
        }
#endif // DOLFINX_NO_STDC_COMPLEX_KERNELS
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
          itg.first->second.emplace_back(id, k, default_facets_ext,
                                         active_coeffs);
        }
        else if (sd != subdomains.end())
        {
          // NOTE: This requires that pairs are sorted
          auto it = std::ranges::lower_bound(sd->second, id, std::less<>{},
                                             [](auto& a) { return a.first; });
          if (it != sd->second.end() and it->first == id)
            itg.first->second.emplace_back(id, k, it->second, active_coeffs);
        }

        if (integral->needs_facet_permutations)
          needs_facet_permutations = true;
      }
    }
  }

  // Attach interior facet kernels
  std::vector<std::int32_t> default_facets_int;
  {
    std::span<const int> ids(ufcx_forms[0].get().form_integral_ids
                                 + integral_offsets[interior_facet],
                             num_integrals_type[interior_facet]);
    auto itg = integrals.insert({IntegralType::interior_facet, {}});
    auto sd = subdomains.find(IntegralType::interior_facet);
    for (std::size_t form_idx = 0; form_idx < ufcx_forms.size(); ++form_idx)
    {
      const ufcx_form& ufcx_form = ufcx_forms[form_idx];
      // Create indicator for interprocess facets
      std::vector<std::int8_t> interprocess_marker;
      if (num_integrals_type[interior_facet] > 0)
      {
        assert(topology->index_map(tdim - 1));
        const std::vector<std::int32_t>& interprocess_facets
            = topology->interprocess_facets();
        std::int32_t num_facets = topology->index_map(tdim - 1)->size_local()
                                  + topology->index_map(tdim - 1)->num_ghosts();
        interprocess_marker.resize(num_facets, 0);
        std::ranges::for_each(interprocess_facets,
                              [&interprocess_marker](auto f)
                              { interprocess_marker[f] = 1; });
      }

      for (int i = 0; i < num_integrals_type[interior_facet]; ++i)
      {
        const int id = ids[i];
        ufcx_integral* integral
            = ufcx_form.form_integrals[integral_offsets[interior_facet] + i];
        assert(integral);
        std::vector<int> active_coeffs;
        for (int j = 0; j < ufcx_form.num_coefficients; ++j)
        {
          if (integral->enabled_coefficients[j])
            active_coeffs.push_back(j);
        }

        kern_t k = nullptr;
        if constexpr (std::is_same_v<T, float>)
          k = integral->tabulate_tensor_float32;
#ifndef DOLFINX_NO_STDC_COMPLEX_KERNELS
        else if constexpr (std::is_same_v<T, std::complex<float>>)
        {
          k = reinterpret_cast<void (*)(
              T*, const T*, const T*,
              const typename scalar_value_type<T>::value_type*, const int*,
              const unsigned char*)>(integral->tabulate_tensor_complex64);
        }
#endif // DOLFINX_NO_STDC_COMPLEX_KERNELS
        else if constexpr (std::is_same_v<T, double>)
          k = integral->tabulate_tensor_float64;
#ifndef DOLFINX_NO_STDC_COMPLEX_KERNELS
        else if constexpr (std::is_same_v<T, std::complex<double>>)
        {
          k = reinterpret_cast<void (*)(
              T*, const T*, const T*,
              const typename scalar_value_type<T>::value_type*, const int*,
              const unsigned char*)>(integral->tabulate_tensor_complex128);
        }
#endif // DOLFINX_NO_STDC_COMPLEX_KERNELS
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
            else if (interprocess_marker[f])
            {
              throw std::runtime_error(
                  "Cannot compute interior facet integral over interprocess "
                  "facet. Please use ghost mode shared facet when creating the "
                  "mesh");
            }
          }
          itg.first->second.emplace_back(id, k, default_facets_int,
                                         active_coeffs);
        }
        else if (sd != subdomains.end())
        {
          auto it = std::ranges::lower_bound(sd->second, id, std::less{},
                                             [](auto& a) { return a.first; });
          if (it != sd->second.end() and it->first == id)
            itg.first->second.emplace_back(id, k, it->second, active_coeffs);
        }

        if (integral->needs_facet_permutations)
          needs_facet_permutations = true;
      }
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
/// @param[in] subdomains Subdomain markers. The data can be computed
/// using fem::compute_integration_domains.
/// @pre Each value in `subdomains` must be sorted by domain id.
/// @param[in] entity_maps The entity maps for the form. Empty for
/// single domain problems.
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
    const std::map<std::shared_ptr<const mesh::Mesh<U>>,
                   std::span<const std::int32_t>>& entity_maps,
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

  return create_form_factory({ufcx_form}, spaces, coeff_map, const_map,
                             subdomains, entity_maps, mesh);
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
/// @param[in] subdomains Subdomain markers. The data can be computed
/// using fem::compute_integration_domains.
/// @pre Each value in `subdomains` must be sorted by domain id.
/// @param[in] entity_maps The entity maps for the form. Empty for
/// single domain problems.
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
    const std::map<std::shared_ptr<const mesh::Mesh<U>>,
                   std::span<const std::int32_t>>& entity_maps,
    std::shared_ptr<const mesh::Mesh<U>> mesh = nullptr)
{
  ufcx_form* form = fptr();
  Form<T, U> L = create_form<T, U>(*form, spaces, coefficients, constants,
                                   subdomains, entity_maps, mesh);
  std::free(form);
  return L;
}

/// @brief NEW Create a function space from a fem::FiniteElement.
template <std::floating_point T>
FunctionSpace<T> create_functionspace(
    std::shared_ptr<mesh::Mesh<T>> mesh,
    std::shared_ptr<const fem::FiniteElement<T>> e,
    std::function<std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>
        reorder_fn
    = nullptr)
{
  assert(e);

  // TODO: check cell type of e (need to add method to fem::FiniteElement)
  assert(mesh);
  assert(mesh->topology());
  if (e->cell_type() != mesh->topology()->cell_type())
    throw std::runtime_error("Cell type of element and mesh must match.");

  // Create element dof layout
  fem::ElementDofLayout layout = fem::create_element_dof_layout(*e);

  // Create a dofmap
  std::function<void(std::span<std::int32_t>, std::uint32_t)> permute_inv
      = e->needs_dof_permutations() ? e->dof_permutation_fn(true, true)
                                    : nullptr;
  auto dofmap = std::make_shared<const DofMap>(create_dofmap(
      mesh->comm(), layout, *mesh->topology(), permute_inv, reorder_fn));

  return FunctionSpace(mesh, e, dofmap);
}

/// @brief Create Expression from UFC
template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
Expression<T, U> create_expression(
    const ufcx_expression& e,
    const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients,
    const std::vector<std::shared_ptr<const Constant<T>>>& constants,
    std::shared_ptr<const FunctionSpace<U>> argument_space = nullptr)
{
  if (e.rank > 0 and !argument_space)
  {
    throw std::runtime_error("Expression has Argument but no Argument "
                             "function space was provided.");
  }

  std::vector<U> X(e.points, e.points + e.num_points * e.entity_dimension);
  std::array<std::size_t, 2> Xshape
      = {static_cast<std::size_t>(e.num_points),
         static_cast<std::size_t>(e.entity_dimension)};
  std::vector<std::size_t> value_shape(e.value_shape,
                                       e.value_shape + e.num_components);
  std::function<void(T*, const T*, const T*,
                     const typename scalar_value_type<T>::value_type*,
                     const int*, const std::uint8_t*)>
      tabulate_tensor = nullptr;
  if constexpr (std::is_same_v<T, float>)
    tabulate_tensor = e.tabulate_tensor_float32;
#ifndef DOLFINX_NO_STDC_COMPLEX_KERNELS
  else if constexpr (std::is_same_v<T, std::complex<float>>)
  {
    tabulate_tensor = reinterpret_cast<void (*)(
        T*, const T*, const T*,
        const typename scalar_value_type<T>::value_type*, const int*,
        const unsigned char*)>(e.tabulate_tensor_complex64);
  }
#endif // DOLFINX_NO_STDC_COMPLEX_KERNELS
  else if constexpr (std::is_same_v<T, double>)
    tabulate_tensor = e.tabulate_tensor_float64;
#ifndef DOLFINX_NO_STDC_COMPLEX_KERNELS
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    tabulate_tensor = reinterpret_cast<void (*)(
        T*, const T*, const T*,
        const typename scalar_value_type<T>::value_type*, const int*,
        const unsigned char*)>(e.tabulate_tensor_complex128);
  }
#endif // DOLFINX_NO_STDC_COMPLEX_KERNELS
  else
    throw std::runtime_error("Type not supported.");

  assert(tabulate_tensor);
  return Expression(coefficients, constants, std::span<const U>(X), Xshape,
                    tabulate_tensor, value_shape, argument_space);
}

/// @brief Create Expression from UFC input (with named coefficients and
/// constants).
template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
Expression<T, U> create_expression(
    const ufcx_expression& e,
    const std::map<std::string, std::shared_ptr<const Function<T, U>>>&
        coefficients,
    const std::map<std::string, std::shared_ptr<const Constant<T>>>& constants,
    std::shared_ptr<const FunctionSpace<U>> argument_space = nullptr)
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

  return create_expression(e, coeff_map, const_map, argument_space);
}

} // namespace dolfinx::fem
