// Copyright (C) 2013-2026 Johan Hake, Jan Blechta, Garth N. Wells and Paul T.
// Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "Form.h"
#include "Function.h"
#include "FunctionSpace.h"
#include "integration_domains.h"
#include "kernel.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/EntityMap.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/utils.h>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <ufcx.h>
#include <utility>
#include <vector>

/// @file form_factory.h
/// @brief Factories for finite element forms from UFCx input.

namespace dolfinx::fem
{
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
template <dolfinx::scalar T, std::floating_point U = scalar_value_t<T>>
Form<T, U> create_form_factory(
    const std::vector<std::reference_wrapper<const ufcx_form>>& ufcx_forms,
    const std::vector<std::shared_ptr<const FunctionSpace<U>>>& spaces,
    const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients,
    const std::vector<std::shared_ptr<const Constant<T>>>& constants,
    const std::map<
        IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>&
        subdomains,
    const std::vector<std::reference_wrapper<const mesh::EntityMap>>&
        entity_maps,
    std::shared_ptr<const mesh::Mesh<U>> mesh = nullptr)
{
  for (const ufcx_form& ufcx_form : ufcx_forms)
  {
    if (ufcx_form.rank != (int)spaces.size())
      throw std::invalid_argument("Wrong number of argument spaces for Form.");
    if (ufcx_form.num_coefficients != (int)coefficients.size())
    {
      throw std::invalid_argument("Mismatch between number of expected and "
                                  "provided Form coefficients.");
    }

    // Check Constants for rank and size consistency
    if (ufcx_form.num_constants != (int)constants.size())
    {
      throw std::invalid_argument(std::format(
          "Mismatch between number of expected and "
          "provided Form Constants. Expected {} constants, but got {}.",
          ufcx_form.num_constants, constants.size()));
    }
    for (std::size_t c = 0; c < constants.size(); ++c)
    {
      if (ufcx_form.constant_ranks[c] != (int)constants[c]->shape.size())
      {
        throw std::invalid_argument(std::format(
            "Mismatch between expected and actual rank of "
            "Form Constant. Rank of Constant {} should be {}, but got rank {}.",
            c, ufcx_form.constant_ranks[c], constants[c]->shape.size()));
      }
      if (!std::equal(constants[c]->shape.begin(), constants[c]->shape.end(),
                      ufcx_form.constant_shapes[c]))
      {
        throw std::invalid_argument(
            std::format("Mismatch between expected and actual shape of Form "
                        "Constant for Constant {}.",
                        c));
      }
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
        throw std::invalid_argument(
            "Cannot create form. Elements are different to "
            "those used to compile the form.");
      }
    }
  }

  // Extract mesh from FunctionSpace, and check they are the same
  if (!mesh and !spaces.empty())
    mesh = spaces.front()->mesh();
  if (!mesh)
    throw std::invalid_argument("No mesh could be associated with the Form.");

  auto topology = mesh->topology();
  assert(topology);
  const int tdim = topology->dim();

  // NOTE: This assumes all forms in mixed-topology meshes have the same
  // integral offsets. Since the UFL forms for each type of cell should be
  // the same, I think this assumption is OK.
  const int* integral_offsets = ufcx_forms[0].get().form_integral_offsets;
  std::array<int, 5> num_integrals_type;
  for (std::size_t i = 0; i < num_integrals_type.size(); ++i)
    num_integrals_type[i] = integral_offsets[i + 1] - integral_offsets[i];

  // Create vertices, if required
  if (num_integrals_type[vertex] > 0)
  {
    mesh->topology_mutable()->create_connectivity(0, tdim);
    mesh->topology_mutable()->create_connectivity(tdim, 0);
  }

  // Create facets, if required
  // NOTE: exterior_facet and interior_facet is declared in ufcx.h
  if (num_integrals_type[exterior_facet] > 0
      or num_integrals_type[interior_facet] > 0)
  {
    mesh->topology_mutable()->create_entities(tdim - 1);
    mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
    mesh->topology_mutable()->create_connectivity(tdim, tdim - 1);
  }

  // Create ridges, if required
  if (num_integrals_type[ridge] > 0)
  {
    mesh->topology_mutable()->create_entities(tdim - 2);
    mesh->topology_mutable()->create_connectivity(tdim - 2, tdim);
    mesh->topology_mutable()->create_connectivity(tdim, tdim - 2);
  }

  // Get list of integral IDs, and load tabulate tensor into memory for
  // each
  std::map<std::tuple<IntegralType, int, int>, integral_data<T, U>> integrals;

  auto check_geometry_hash
      = [&geo = mesh->geometry()](const ufcx_integral& integral,
                                  std::size_t cell_idx)
  {
    if (integral.coordinate_element_hash != geo.cmaps().at(cell_idx).hash())
    {
      throw std::runtime_error(std::format(
          "Generated integral geometry element does not match mesh geometry: "
          "{}, {}",
          integral.coordinate_element_hash, geo.cmaps().at(cell_idx).hash()));
    }
  };

  // Attach cell kernels
  bool needs_facet_permutations = false;
  {
    std::vector<std::int32_t> default_cells;
    std::span<const int> ids(ufcx_forms[0].get().form_integral_ids
                                 + integral_offsets[cell],
                             num_integrals_type[cell]);
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
        check_geometry_hash(*integral, form_idx);

        // Build list of active coefficients
        std::vector<int> active_coeffs;
        for (int j = 0; j < ufcx_form.num_coefficients; ++j)
        {
          if (integral->enabled_coefficients[j])
            active_coeffs.push_back(j);
        }

        impl::kernel_t<T, U> k = impl::extract_kernel<T, U>(integral);
        if (!k)
        {
          throw std::invalid_argument(
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
          integrals.insert({{IntegralType::cell, i, form_idx},
                            {k, default_cells, active_coeffs}});
        }
        else if (sd != subdomains.end())
        {
          // NOTE: This requires that pairs are sorted
          auto it = std::ranges::lower_bound(sd->second, id, std::less<>{},
                                             [](auto& a) { return a.first; });
          if (it != sd->second.end() and it->first == id)
          {
            integrals.insert({{IntegralType::cell, i, form_idx},
                              {k,
                               std::vector<std::int32_t>(it->second.begin(),
                                                         it->second.end()),
                               active_coeffs}});
          }
        }

        if (integral->needs_facet_permutations)
          needs_facet_permutations = true;
      }
    }
  }

  // Attach interior facet kernels
  {
    std::vector<std::int32_t> default_facets_int;
    std::span<const int> ids(ufcx_forms[0].get().form_integral_ids
                                 + integral_offsets[interior_facet],
                             num_integrals_type[interior_facet]);
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
        check_geometry_hash(*integral, form_idx);

        std::vector<int> active_coeffs;
        for (int j = 0; j < ufcx_form.num_coefficients; ++j)
        {
          if (integral->enabled_coefficients[j])
            active_coeffs.push_back(j);
        }

        impl::kernel_t<T, U> k = impl::extract_kernel<T, U>(integral);
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
              std::array<std::int32_t, 4> pairs
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
          integrals.insert({{IntegralType::interior_facet, i, form_idx},
                            {k, default_facets_int, active_coeffs}});
        }
        else if (sd != subdomains.end())
        {
          auto it = std::ranges::lower_bound(sd->second, id, std::less{},
                                             [](auto& a) { return a.first; });
          if (it != sd->second.end() and it->first == id)
          {
            integrals.insert({{IntegralType::interior_facet, i, form_idx},
                              {k,
                               std::vector<std::int32_t>(it->second.begin(),
                                                         it->second.end()),
                               active_coeffs}});
          }
        }

        if (integral->needs_facet_permutations)
          needs_facet_permutations = true;
      }
    }
  }

  // Attach exterior entity integrals
  {
    for (IntegralType itg_type : {IntegralType::exterior_facet,
                                  IntegralType::vertex, IntegralType::ridge})
    {
      std::size_t dim;
      switch (itg_type)
      {
      case IntegralType::exterior_facet:
      {
        dim = tdim - 1;
        break;
      }
      case IntegralType::ridge:
      {
        dim = tdim - 2;
        break;
      }
      case IntegralType::vertex:
      {
        dim = 0;
        break;
      }
      default:
        throw std::invalid_argument("Unsupported integral type");
      }

      const std::function<std::vector<std::int32_t>(const mesh::Topology&,
                                                    IntegralType)>
          get_default_integration_entities
          = [dim](const mesh::Topology& topology, IntegralType itype)
      {
        if (itype == IntegralType::exterior_facet)
        {
          // Integrate over all owned exterior facets
          return mesh::exterior_facet_indices(topology);
        }
        else
        {
          // Integrate over all owned entities
          std::int32_t num_entities = topology.index_map(dim)->size_local();
          std::vector<std::int32_t> entities(num_entities);
          std::iota(entities.begin(), entities.end(), 0);
          return entities;
        }
      };

      std::vector<std::int32_t> default_entities_ext;

      std::span<const int> ids(ufcx_forms[0].get().form_integral_ids
                                   + integral_offsets[(std::int8_t)itg_type],
                               num_integrals_type[(std::int8_t)itg_type]);
      auto sd = subdomains.find(itg_type);
      for (std::size_t form_idx = 0; form_idx < ufcx_forms.size(); ++form_idx)
      {
        const ufcx_form& ufcx_form = ufcx_forms[form_idx];
        for (int i = 0; i < num_integrals_type[(std::int8_t)itg_type]; ++i)
        {
          const int id = ids[i];
          ufcx_integral* integral
              = ufcx_form.form_integrals[integral_offsets[(std::int8_t)itg_type]
                                         + i];
          assert(integral);
          check_geometry_hash(*integral, form_idx);

          std::vector<int> active_coeffs;
          for (int j = 0; j < ufcx_form.num_coefficients; ++j)
          {
            if (integral->enabled_coefficients[j])
              active_coeffs.push_back(j);
          }

          impl::kernel_t<T, U> k = impl::extract_kernel<T, U>(integral);

          // Build list of entities to assembler over
          auto e_to_c = topology->connectivity(dim, tdim);
          assert(e_to_c);
          auto c_to_e = topology->connectivity(tdim, dim);
          assert(c_to_e);
          if (id == -1)
          {
            std::vector default_entities
                = get_default_integration_entities(*topology, itg_type);
            // Default kernel
            default_entities_ext.reserve(2 * default_entities.size());
            for (std::int32_t e : default_entities)
            {
              // There will only be one pair for an exterior facet integral
              std::array<std::int32_t, 2> pair = impl::get_cell_entity_pairs<1>(
                  e, e_to_c->links(e), *c_to_e);
              default_entities_ext.insert(default_entities_ext.end(),
                                          pair.begin(), pair.end());
            }
            integrals.insert({{itg_type, i, form_idx},
                              {k, default_entities_ext, active_coeffs}});
          }
          else if (sd != subdomains.end())
          {
            // NOTE: This requires that pairs are sorted
            auto it = std::ranges::lower_bound(sd->second, id, std::less<>{},
                                               [](auto& a) { return a.first; });
            if (it != sd->second.end() and it->first == id)
            {
              integrals.insert({{itg_type, i, form_idx},
                                {k,
                                 std::vector<std::int32_t>(it->second.begin(),
                                                           it->second.end()),
                                 active_coeffs}});
            }
          }

          if (integral->needs_facet_permutations)
            needs_facet_permutations = true;
        }
      }
    }
  }

  return Form<T, U>(spaces, std::move(integrals), mesh, coefficients, constants,
                    needs_facet_permutations, entity_maps);
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
template <dolfinx::scalar T, std::floating_point U = scalar_value_t<T>>
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
    const std::vector<std::reference_wrapper<const mesh::EntityMap>>&
        entity_maps,
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
      throw std::runtime_error(
          std::format("Form coefficient \"{}\" not provided.", name));
    }
  }

  // Place constants in appropriate order
  std::vector<std::shared_ptr<const Constant<T>>> const_map;
  for (const std::string& name : get_constant_names(ufcx_form))
  {
    if (auto it = constants.find(name); it != constants.end())
      const_map.push_back(it->second);
    else
      throw std::runtime_error(
          std::format("Form constant \"{}\" not provided.", name));
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
template <dolfinx::scalar T, std::floating_point U = scalar_value_t<T>>
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
    const std::vector<std::reference_wrapper<const mesh::EntityMap>>&
        entity_maps,
    std::shared_ptr<const mesh::Mesh<U>> mesh = nullptr)
{
  ufcx_form* form = fptr();
  Form<T, U> L = create_form<T, U>(*form, spaces, coefficients, constants,
                                   subdomains, entity_maps, mesh);
  std::free(form);
  return L;
}
} // namespace dolfinx::fem
