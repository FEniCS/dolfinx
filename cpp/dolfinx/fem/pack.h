// Copyright (C) 2013-2026 Garth N. Wells and Jørgen S. Dokken
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
#include <format>
#include <ranges>
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
  std::size_t num_entities = 0;
  int cstride = 0;
  if (const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients
      = form.coefficients();
      !coefficients.empty())
  {
    const std::vector<int> offsets = form.coefficient_offsets();
    cstride = offsets.back();
    num_entities = form.domain(integral_type, id, 0).size();
    if (integral_type != IntegralType::cell)
      num_entities /= 2;
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
  for (fem::IntegralType type : form.integral_types())
  {
    for (int i = 0; i < form.num_integrals(type, 0); ++i)
    {
      coeffs.emplace_hint(coeffs.end(), std::pair{type, i},
                          allocate_coefficient_storage(form, type, i));
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
  const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients
      = form.coefficients();
  const std::vector<int> offsets = form.coefficient_offsets();

  for (auto& [intergal_data, coeff_data] : coeffs)
  {
    auto [integral_type, id] = intergal_data;
    std::vector<T>& c = coeff_data.first;
    int cstride = coeff_data.second;
    if (!coefficients.empty())
    {
      switch (integral_type)
      {
      case IntegralType::cell:
      {
        // Iterate over coefficients that are active in cell integrals
        for (int coeff : form.active_coeffs(IntegralType::cell, id))
        {
          // Get coefficient mesh
          auto mesh = coefficients[coeff]->function_space()->mesh();
          assert(mesh);

          // Other integrals in the form might have coefficients defined
          // over entities of codim > 0, which don't make sense for cell
          // integrals, so don't pack them.
          if (int codim
              = form.mesh()->topology()->dim() - mesh->topology()->dim();
              codim > 0)
          {
            throw std::runtime_error("Should not be packing coefficients with "
                                     "codim>0 in a cell integral");
          }

          std::span<const std::int32_t> cells_b
              = form.domain_coeff(IntegralType::cell, id, coeff);
          md::mdspan cells(cells_b.data(), cells_b.size());
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
        // Iterate over coefficients that are active in interior
        // facet integrals
        for (int coeff : form.active_coeffs(IntegralType::interior_facet, id))
        {
          auto mesh = coefficients[coeff]->function_space()->mesh();
          std::span<const std::int32_t> facets_b
              = form.domain_coeff(IntegralType::interior_facet, id, coeff);
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
      case IntegralType::exterior_facet:
      case IntegralType::vertex:
      case IntegralType::ridge:
      {
        // Iterate over coefficients that are active in vertex integrals
        for (int coeff : form.active_coeffs(integral_type, id))
        {
          // Get coefficient mesh
          auto mesh = coefficients[coeff]->function_space()->mesh();
          assert(mesh);

          std::span<const std::int32_t> entitites_b
              = form.domain_coeff(integral_type, id, coeff);
          md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 2>>
              entities(entitites_b.data(), entitites_b.size() / 2, 2);
          std::span<const std::uint32_t> cell_info
              = impl::get_cell_orientation_info(*coefficients[coeff]);
          impl::pack_coefficient_entity(
              std::span(c), cstride, *coefficients[coeff], cell_info,
              md::submdspan(entities, md::full_extent, 0), offsets[coeff]);
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

/// @brief Given a Function and a related mesh and its integration entities,
/// extract the cell indices of the coefficient mesh.
/// @param[in] coeff The coefficient to extract cell indices for.
/// @param[in] mesh The mesh which the integration entities belong to.
/// @param[in] entities The integration entities. Is either a sequence of local
/// cell indices, or a sequence of (cell, local entity index) tuples.
/// @param[in] entity_map The map between the mesh and the coefficient mesh in
/// case they are different.
/// @returns A vector of cell indices on the coefficient mesh corresponding to
/// the integration entities.
template <dolfinx::scalar T, std::floating_point U>
std::vector<std::int32_t> extract_coefficient_cells_from_entities(
    const fem::Function<T, U>& coeff, const mesh::Mesh<U>& mesh,
    fem::MDSpan2 auto entities,
    std::optional<std::reference_wrapper<const dolfinx::mesh::EntityMap>>
        entity_map)
{
  auto mesh_c = coeff.function_space()->mesh();
  assert(mesh_c);

  auto span_to_vector = [](auto entities)
  {
    assert(entities.rank() == 1);

    std::vector<std::int32_t> vec;
    vec.reserve(entities.extent(0));
    for (std::size_t i = 0; i < entities.extent(0); ++i)
      vec.push_back(entities[i]);
    return vec;
  };

  if (mesh_c->topology() == mesh.topology())
  {
    // If same mesh no mapping is needed
    if constexpr (entities.rank() == 1)
      return span_to_vector(entities);

    else
      // If (cell, local_index) pairs are given, extract the cells
      return span_to_vector(md::submdspan(entities, md::full_extent, 0));
  }
  else
  {
    assert(entity_map.has_value());
    const mesh::Topology& topology = *mesh.topology();
    int tdim = topology.dim();
    int codim = tdim - mesh_c->topology()->dim();
    const dolfinx::mesh::EntityMap& emap = entity_map.value().get();
    bool inverse = emap.sub_topology() == mesh_c->topology();
    // If cells are supplied on the parent mesh, we can directly map them to
    // cells on the coefficient mesh.
    if constexpr (entities.rank() == 1)
    {
      assert(codim == 0);

      return emap.sub_topology_to_topology(span_to_vector(entities), inverse);
    }
    else if constexpr (entities.rank() == 2)
    {
      if (codim == 0)
      {
        // If codim is zero we extract the cells and map them
        auto cells = md::submdspan(entities, md::full_extent, 0);
        return emap.sub_topology_to_topology(span_to_vector(cells), inverse);
      }
      else
      {
        // Any other codim needs  to map (cell, local index) to facets and then
        // to cells of the submesh
        if (!inverse)
        {
          throw std::runtime_error(
              "Unsupported mapping. Can only map from submesh to parent mesh.");
        }
        assert(codim > 0);
        auto c_to_e = topology.connectivity(tdim, tdim - codim);
        if (!c_to_e)
        {
          throw std::runtime_error(std::format(
              "Topology connectivity from codim {} to {} not found.", tdim,
              tdim - codim));
        }
        // Map parent (cell, local_index) to parent facet
        std::vector<std::int32_t> contiguous_cells;
        contiguous_cells.reserve(entities.extent(0));
        for (std::size_t e = 0; e < entities.extent(0); ++e)
        {
          auto pair = md::submdspan(entities, e, md::full_extent);
          contiguous_cells.push_back(c_to_e->links(pair[0])[pair[1]]);
        }
        // Map parent facet to submesh cell
        return emap.sub_topology_to_topology(contiguous_cells, inverse);
      }
    }
  }
}

/// @brief Pack coefficient data over a list of cells or facets.
///
/// Typically used to prepare coefficient data for an ::Expression.
/// @tparam T Data type of coefficients
/// @tparam U Floating point type of mesh geometry
/// @param coeffs Coefficients to pack
/// @param mesh Mesh which the entities belong to
/// @param entities Entities to pack over
/// @param entity_maps Bidirectional maps between the entities of a
/// parent mesh and a submesh in case of coefficients being defined on
/// both.
/// @param offsets Insertion offset for each of the `coeffs` when packed
/// into `c`.
/// @param[in,out] c Packed coefficients.
template <dolfinx::scalar T, std::floating_point U>
void pack_coefficients(
    std::vector<std::reference_wrapper<const Function<T, U>>> coeffs,
    const mesh::Mesh<U>& mesh, fem::MDSpan2 auto entities,
    const std::vector<std::reference_wrapper<const dolfinx::mesh::EntityMap>>&
        entity_maps,
    std::span<const int> offsets, std::span<T> c)
{

  assert(!offsets.empty());
  const int cstride = offsets.back();

  if (c.size() < entities.extent(0) * offsets.back())
    throw std::runtime_error("Coefficient packing span is too small.");

  // Iterate over coefficients
  for (std::size_t coeff = 0; coeff < coeffs.size(); ++coeff)
  {

    // Get mesh of coefficient and check if entity map is required
    auto mesh_c = coeffs[coeff].get().function_space()->mesh();
    std::vector<std::int32_t> coefficient_cells;
    if (mesh_c->topology() == mesh.topology())
    {
      coefficient_cells = extract_coefficient_cells_from_entities(
          coeffs[coeff].get(), mesh, entities, std::nullopt);
    }
    else
    {
      // Helper function to get correct entity map
      auto get_entity_map
          = [mesh, &entity_maps](auto& mesh0) -> const mesh::EntityMap&
      {
        auto it = std::ranges::find_if(
            entity_maps,
            [mesh, mesh0](const mesh::EntityMap& em)
            {
              return ((em.topology() == mesh0->topology()
                       and em.sub_topology() == mesh.topology()))
                     or ((em.sub_topology() == mesh0->topology()
                          and em.topology() == mesh.topology()));
            });

        if (it == entity_maps.end())
        {
          throw std::runtime_error("Incompatible mesh. argument "
                                   "entity_maps must be provided.");
        }
        return *it;
      };
      // Find correct entity map and determine direction of the map
      const mesh::EntityMap& emap = get_entity_map(mesh_c);
      coefficient_cells = extract_coefficient_cells_from_entities(
          coeffs[coeff].get(), mesh, entities,
          std::reference_wrapper<const mesh::EntityMap>(emap));
    }

    std::span<const std::uint32_t> cell_info
        = impl::get_cell_orientation_info(coeffs[coeff].get());
    md::mdspan cells(coefficient_cells.data(), coefficient_cells.size());
    impl::pack_coefficient_entity(std::span(c), cstride, coeffs[coeff].get(),
                                  cell_info, cells, offsets[coeff]);
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
    std::ranges::copy(constant.get().value,
                      std::next(constant_values.begin(), offset));
    offset += constant.get().value.size();
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
