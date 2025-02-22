// Copyright (C) 2013-2025 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "DofMap.h"
#include "FiniteElement.h"
#include "Form.h"
#include "Function.h"
#include "FunctionSpace.h"
#include "traits.h"
#include <array>
#include <basix/mdspan.hpp>
#include <concepts>
#include <dolfinx/mesh/Topology.h>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

/// @file pack.h
/// @brief Functions supporting the packing of coefficient data.

namespace dolfinx::fem
{
template <dolfinx::scalar T, std::floating_point U>
class Expression;

namespace impl
{
/// @private
template <dolfinx::scalar T, std::floating_point U>
std::span<const std::uint32_t>
get_cell_orientation_info(const Function<T, U>& coefficient)
{
  std::span<const std::uint32_t> cell_info;
  auto element = coefficient.function_space()->element();
  assert(element);
  if (element->needs_dof_transformations())
  {
    auto mesh = coefficient.function_space()->mesh();
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  return cell_info;
}

/// Pack a single coefficient for a single cell
template <int _bs, dolfinx::scalar T>
void pack_impl(std::span<T> coeffs, std::int32_t cell, int bs,
               std::span<const T> v, std::span<const std::uint32_t> cell_info,
               const DofMap& dofmap, auto transform)
{
  std::span<const std::int32_t> dofs = dofmap.cell_dofs(cell);
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
      assert(_bs == bs);
      const int pos_c = _bs * i;
      const int pos_v = _bs * dofs[i];
      for (int k = 0; k < _bs; ++k)
        coeffs[pos_c + k] = v[pos_v + k];
    }
  }

  transform(coeffs, cell_info, cell, 1);
}

/// @brief Pack a single coefficient for a set of active entities.
///
/// @param[out] c Coefficient to be packed.
/// @param[in] cstride Total number of coefficient values to pack for
/// each entity.
/// @param[in] u Function to extract coefficient data from.
/// @param[in] cell_info Array of bytes describing which transformation
/// has to be applied on the cell to map it to the reference element.
/// @param[in] cells Set of active cells.
/// @param[in] offset The offset for c.
template <dolfinx::scalar T, std::floating_point U>
void pack_coefficient_entity(std::span<T> c, int cstride,
                             const Function<T, U>& u,
                             std::span<const std::uint32_t> cell_info,
                             auto cells, std::int32_t offset)
{
  static_assert(cells.rank() == 1);

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
    for (std::size_t e = 0; e < cells.extent(0); ++e)
    {
      if (std::int32_t cell = cells(e); cell >= 0)
      {
        auto cell_coeff = c.subspan(e * cstride + offset, space_dim);
        pack_impl<1>(cell_coeff, cell, bs, v, cell_info, dofmap,
                     transformation);
      }
    }
    break;
  case 2:
    for (std::size_t e = 0; e < cells.extent(0); ++e)
    {
      if (std::int32_t cell = cells(e); cell >= 0)
      {
        auto cell_coeff = c.subspan(e * cstride + offset, space_dim);
        pack_impl<2>(cell_coeff, cell, bs, v, cell_info, dofmap,
                     transformation);
      }
    }
    break;
  case 3:
    for (std::size_t e = 0; e < cells.extent(0); ++e)
    {
      if (std::int32_t cell = cells(e); cell >= 0)
      {
        auto cell_coeff = c.subspan(e * cstride + offset, space_dim);
        pack_impl<3>(cell_coeff, cell, bs, v, cell_info, dofmap,
                     transformation);
      }
    }
    break;
  default:
    for (std::size_t e = 0; e < cells.extent(0); ++e)
    {
      if (std::int32_t cell = cells(e); cell >= 0)
      {
        auto cell_coeff = c.subspan(e * cstride + offset, space_dim);
        pack_impl<-1>(cell_coeff, cell, bs, v, cell_info, dofmap,
                      transformation);
      }
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
    num_entities = form.domain(integral_type, id, 0).size();
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

/// @brief Pack coefficients of a Form.
///
/// @param[in] form Form to pack the coefficients for.
/// @param[in,out] coeffs Map from a `(integral_type, domain_id)` pair
/// to a `(coeffs, cstride)` pair.
/// - `coeffs` is an array of shape `(num_int_entities, cstride)` into
/// which coefficient data will be packed.
/// - `num_int_entities` is the number of entities over which
/// coefficient data is packed.
/// - `cstride` is the number of coefficient data entries per entity.
/// - `coeffs` is flattened using  row-major layout.
template <dolfinx::scalar T, std::floating_point U>
void pack_coefficients(const Form<T, U>& form,
                       std::map<std::pair<IntegralType, int>,
                                std::pair<std::vector<T>, int>>& coeffs)
{
  namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

  const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients
      = form.coefficients();
  const std::vector<int> offsets = form.coefficient_offsets();

  for (auto& [intergal_data, coeff_data] : coeffs)
  {
    IntegralType integral_type = intergal_data.first;
    int id = intergal_data.second;
    std::vector<T>& c = coeff_data.first;
    int cstride = coeff_data.second;

    // Indicator for packing coefficients
    std::vector<int> active_coefficient(coefficients.size(), 0);
    if (!coefficients.empty())
    {
      switch (integral_type)
      {
      case IntegralType::cell:
      {
        // Get indicator for all coefficients that are active in cell
        // integrals
        for (std::size_t i = 0; i < form.num_integrals(IntegralType::cell); ++i)
        {
          for (auto idx : form.active_coeffs(IntegralType::cell, i))
            active_coefficient[idx] = 1;
        }

        // Iterate over coefficients
        for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
        {
          if (!active_coefficient[coeff])
            continue;

          // Get coefficient mesh
          auto mesh = coefficients[coeff]->function_space()->mesh();
          assert(mesh);

          // Other integrals in the form might have coefficients defined
          // over entities of codim > 0, which don't make sense for cell
          // integrals, so don't pack them.
          if (const int codim
              = form.mesh()->topology()->dim() - mesh->topology()->dim();
              codim > 0)
          {
            throw std::runtime_error("Should not be packing coefficients with "
                                     "codim>0 in a cell integral");
          }

          const std::vector<std::int32_t> cells_b
              = form.xdomain(IntegralType::cell, id, 0, *mesh);
          md::mdspan cells(cells_b.data(), cells_b.size());
          std::span<const std::uint32_t> cell_info
              = impl::get_cell_orientation_info(*coefficients[coeff]);
          impl::pack_coefficient_entity(std::span(c), cstride,
                                        *coefficients[coeff], cell_info, cells,
                                        offsets[coeff]);
        }
        break;
      }
      case IntegralType::exterior_facet:
      {
        // Get indicator for all coefficients that are active in
        // exterior facet integrals
        for (std::size_t i = 0;
             i < form.num_integrals(IntegralType::exterior_facet); ++i)
        {
          for (auto idx : form.active_coeffs(IntegralType::exterior_facet, i))
            active_coefficient[idx] = 1;
        }

        // Iterate over coefficients
        for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
        {
          if (!active_coefficient[coeff])
            continue;

          auto mesh = coefficients[coeff]->function_space()->mesh();
          const std::vector<std::int32_t> facets_b
              = form.xdomain(IntegralType::exterior_facet, id, 0, *mesh);
          md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 2>>
              facets(facets_b.data(), facets_b.size() / 2, 2);
          auto cells = md::submdspan(facets, md::full_extent, 0);

          std::span<const std::uint32_t> cell_info
              = impl::get_cell_orientation_info(*coefficients[coeff]);
          impl::pack_coefficient_entity(std::span(c), cstride,
                                        *coefficients[coeff], cell_info, cells,
                                        offsets[coeff]);
        }
        break;
      }
      case IntegralType::interior_facet:
      {
        // Get indicator for all coefficients that are active in interior
        // facet integrals
        for (std::size_t i = 0;
             i < form.num_integrals(IntegralType::interior_facet); ++i)
        {
          for (auto idx : form.active_coeffs(IntegralType::interior_facet, i))
            active_coefficient[idx] = 1;
        }

        // Iterate over coefficients
        for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
        {
          if (!active_coefficient[coeff])
            continue;

          auto mesh = coefficients[coeff]->function_space()->mesh();
          const std::vector<std::int32_t> facets_b
              = form.xdomain(IntegralType::interior_facet, id, 0, *mesh);
          md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 4>>
              facets(facets_b.data(), facets_b.size() / 4, 4);

          std::span<const std::uint32_t> cell_info
              = impl::get_cell_orientation_info(*coefficients[coeff]);

          // Pack coefficient ['+']
          auto cells0 = md::submdspan(facets, md::full_extent, 0);
          impl::pack_coefficient_entity(std::span(c), 2 * cstride,
                                        *coefficients[coeff], cell_info, cells0,
                                        2 * offsets[coeff]);

          // Pack coefficient ['-']
          auto cells1 = md::submdspan(facets, md::full_extent, 2);
          impl::pack_coefficient_entity(std::span(c), 2 * cstride,
                                        *coefficients[coeff], cell_info, cells1,
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
}

/// @brief Pack coefficient data over a list of cells or facets.
///
/// Typically used to prepare coefficient data for an ::Expression.
/// @tparam T
/// @tparam U
/// @param coeffs Coefficients to pack
/// @param offsets Offsets
/// @param entities Entities to pack over
/// @param[out] c Packed coefficients.
template <dolfinx::scalar T, std::floating_point U>
void pack_coefficients(
    std::vector<std::reference_wrapper<const Function<T, U>>> coeffs,
    std::span<const int> offsets, fem::MDSpan2 auto entities, std::span<T> c)
{
  namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

  assert(!offsets.empty());
  const int cstride = offsets.back();

  if (c.size() < entities.extent(0) * offsets.back())
    throw std::runtime_error("Coefficient packing span is too small.");

  // Iterate over coefficients
  for (std::size_t coeff = 0; coeff < coeffs.size(); ++coeff)
  {
    std::span<const std::uint32_t> cell_info
        = impl::get_cell_orientation_info(coeffs[coeff].get());

    if constexpr (entities.rank() == 1)
    {
      impl::pack_coefficient_entity(std::span(c), cstride, coeffs[coeff].get(),
                                    cell_info, entities, offsets[coeff]);
    }
    else
    {
      auto cells = md::submdspan(entities, md::full_extent, 0);
      impl::pack_coefficient_entity(std::span(c), cstride, coeffs[coeff].get(),
                                    cell_info, cells, offsets[coeff]);
    }
  }
}

/// @brief Pack constants of an Expression or Form into a single array
/// ready for assembly.
/// @param c Constants to pack.
/// @return Packed constants
template <typename T>
std::vector<T>
pack_constants(std::vector<std::reference_wrapper<const fem::Constant<T>>> c)
{
  // Calculate size of array needed to store packed constants
  std::int32_t size = std::accumulate(
      c.cbegin(), c.cend(), 0, [](std::int32_t sum, auto& constant)
      { return sum + constant.get().value.size(); });

  // Pack constants
  std::vector<T> constant_values(size);
  std::int32_t offset = 0;
  for (auto& constant : c)
  {
    const std::vector<T>& value = constant.get().value;
    std::ranges::copy(value, std::next(constant_values.begin(), offset));
    offset += value.size();
  }

  return constant_values;
}

/// @brief Pack constants of an Expression or Form into a single array
/// ready for assembly.
/// @param u The Expression or Form to pack constant data for.
/// @return Packed constants
template <typename U>
  requires std::convertible_to<
               U, fem::Expression<typename std::decay_t<U>::scalar_type,
                                  typename std::decay_t<U>::geometry_type>>
           or std::convertible_to<
               U, fem::Form<typename std::decay_t<U>::scalar_type,
                            typename std::decay_t<U>::geometry_type>>
std::vector<typename U::scalar_type> pack_constants(const U& u)
{
  using T = typename std::decay_t<U>::scalar_type;
  std::vector<std::reference_wrapper<const Constant<T>>> c;
  std::ranges::transform(u.constants(), std::back_inserter(c),
                         [](auto c) -> const Constant<T>& { return *c; });
  return fem::pack_constants(c);
}

} // namespace dolfinx::fem
