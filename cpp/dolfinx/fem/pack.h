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
/// @brief Get cell permutation data for a coefficient, if its element
/// needs it, otherwise an empty span.
/// @param[in] coefficient Coefficient to get cell permutation data for.
/// @return Cell permutation data, indexed by cell (see
/// mesh::Topology::get_cell_permutation_info), or an empty span if
/// `coefficient`'s element does not require it.
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

/// @brief Gather a single coefficient's degrees-of-freedom for a single
/// cell and apply its DOF transformation.
/// @param[out] coeffs Destination for this cell's packed values.
/// @param[in] cell Cell to gather DOF values for.
/// @param[in] bs Block size of `dofmap`, as a run-time `int` or a
/// compile-time `std::integral_constant<int, N>` (allows the compiler
/// to specialize and unroll the inner loop for the common block sizes
/// 1, 2 and 3).
/// @param[in] v Function values to gather from, indexed by
/// process-local DOF (see DofMap::cell_dofs).
/// @param[in] cell_info Cell permutation data (see
/// get_cell_orientation_info), passed through to `transform`.
/// @param[in] dofmap Dofmap used to look up `cell`'s DOFs.
/// @param[in] transform DOF transformation applied to `coeffs` after
/// gathering (see FiniteElement::dof_transformation_fn).
template <dolfinx::scalar T>
void pack_impl(std::span<T> coeffs, std::int32_t cell, auto bs,
               std::span<const T> v, std::span<const std::uint32_t> cell_info,
               const DofMap& dofmap, auto transform)
{
  std::span<const std::int32_t> dofs = dofmap.cell_dofs(cell);
  for (std::size_t i = 0; i < dofs.size(); ++i)
  {
    const int pos_c = bs * i;
    const int pos_v = bs * dofs[i];
    for (int k = 0; k < bs; ++k)
      coeffs[pos_c + k] = v[pos_v + k];
  }

  if (transform)
    transform(coeffs, cell_info, cell, 1);
}

/// @brief Pack a single coefficient for a set of active entities.
///
/// @tparam T Scalar type of the coefficient.
/// @tparam U Floating point type of the mesh geometry.
/// @param[out] c Coefficient storage to pack into, shape
/// `(cells.extent(0), cstride)`, flattened row-major.
/// @param[in] cstride Row length of `c`, i.e. the total number of
/// coefficient values packed per entity across *all* coefficients
/// sharing `c` (not just `u`'s own `space_dim` values).
/// @param[in] u Function to extract coefficient data from.
/// @param[in] cell_info Cell permutation information, indexed by cell
/// (see get_cell_orientation_info).
/// @param[in] cells Cell index for each active entity. A negative
/// entry marks an entity absent from `u`'s mesh (e.g. the far side of
/// an interface), and is skipped.
/// @param[in] offset Offset of `u`'s data within each row of `c`.
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

  const int bs = dofmap.bs();

  // Transformation from conforming degrees-of-freedom to reference
  // degrees-of-freedom. Use whichever of DirectDofTransform or the
  // generic std::function-based closure applies -- see
  // FiniteElement::with_dof_transformation_fn for the rationale.
  element->template with_dof_transformation_fn<T, doftransform::transpose>(
      [&cells, &c, &cstride, &offset, &space_dim, &bs, &v, &cell_info,
       &dofmap](const auto& transformation)
      {
        // Passing the block size as a compile-time constant
        // (std::integral_constant<int, N>) lets `pack_impl` unroll its
        // inner (per-DOF) loop for the common block sizes 1, 2, and 3,
        // rather than looping `bs` times at runtime for every cell.
        auto pack_for_bs = [&cells, &c, &cstride, &offset, &space_dim, &v,
                            &cell_info, &dofmap, &transformation](auto bs)
        {
          for (std::size_t e = 0; e < cells.extent(0); ++e)
          {
            if (std::int32_t cell = cells(e); cell >= 0)
            {
              auto cell_coeff = c.subspan(e * cstride + offset, space_dim);
              pack_impl(cell_coeff, cell, bs, v, cell_info, dofmap,
                        transformation);
            }
          }
        };

        switch (bs)
        {
        case 1:
          pack_for_bs(std::integral_constant<int, 1>());
          break;
        case 2:
          pack_for_bs(std::integral_constant<int, 2>());
          break;
        case 3:
          pack_for_bs(std::integral_constant<int, 3>());
          break;
        default:
          pack_for_bs(bs);
          break;
        }
      });
}
} // namespace impl

/// @brief Allocate storage for coefficients of a pair `(integral_type,
/// idx)` from a Form.
/// @param[in] form The Form
/// @param[in] integral_type Type of integral
/// @param[in] idx Integral index in the flattened list of integral
/// kernels (see Form::domain).
/// @return A storage container and the column stride
template <dolfinx::scalar T, std::floating_point U>
std::pair<std::vector<T>, int>
allocate_coefficient_storage(const Form<T, U>& form, IntegralType integral_type,
                             int idx)
{
  std::size_t num_entities = 0;
  int cstride = 0;
  if (const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients
      = form.coefficients();
      !coefficients.empty())
  {
    const std::vector<int> offsets = form.coefficient_offsets();
    cstride = offsets.back();

    // `domain()` returns entities flattened as (cell,) for cell
    // integrals, (cell, local_entity_index) pairs for exterior_facet/
    // vertex/ridge integrals, and (cell, local_facet, cell,
    // local_facet) quadruples for interior_facet integrals (one '+'
    // and one '-' side). Dividing by 2 therefore gives the number of
    // entities for exterior_facet/vertex/ridge integrals, but *twice*
    // the number of facets for interior_facet integrals -- which is
    // exactly the entity count required, since interior_facet
    // coefficient data is packed at a doubled `cstride` (one side
    // each), see ::pack_coefficients.
    num_entities = form.domain(integral_type, idx, 0).size();
    if (integral_type != IntegralType::cell)
      num_entities /= 2;
  }

  return {std::vector<T>(num_entities * cstride), cstride};
}

/// @brief Allocate memory for packed coefficients of a Form.
/// @param[in] form The Form
/// @return Map from a form `(integral_type, idx)` pair to a `(coeffs,
/// cstride)` pair, where `idx` is the integral index in the
/// flattened list of integral kernels (see Form::domain).
template <dolfinx::scalar T, std::floating_point U>
std::map<std::pair<IntegralType, int>, std::pair<std::vector<T>, int>>
allocate_coefficient_storage(const Form<T, U>& form)
{
  std::map<std::pair<IntegralType, int>, std::pair<std::vector<T>, int>> coeffs;
  for (fem::IntegralType type : form.integral_types())
  {
    // `num_integrals` scans all of `form`'s integrals, so it is
    // evaluated once per `type` here rather than as the loop
    // condition (which would re-scan on every iteration).
    const int n = form.num_integrals(type, 0);
    for (int idx = 0; idx < n; ++idx)
    {
      coeffs.emplace_hint(coeffs.end(), std::pair{type, idx},
                          allocate_coefficient_storage(form, type, idx));
    }
  }

  return coeffs;
}

/// @brief Pack coefficients of a Form.
///
/// @param[in] form Form to pack the coefficients for.
/// @param[in,out] coeffs Map from a `(integral_type, idx)` pair, where
/// `idx` is the integral index in the flattened list of integral
/// kernels (see Form::domain), to a `(coeffs, cstride)` pair, as
/// returned by ::allocate_coefficient_storage.
/// - `coeffs` is an array of shape `(num_int_entities, cstride)` into
/// which coefficient data will be packed.
/// - `num_int_entities` is the number of entities over which
/// coefficient data is packed.
/// - `cstride` is the number of coefficient data entries per entity.
/// - `coeffs` is flattened using  row-major layout.
///
/// @note The `(num_int_entities, cstride)` shape above holds as stated
/// for `IntegralType::cell`, `exterior_facet`, `vertex`, and `ridge`
/// integrals. For `IntegralType::interior_facet`, each facet's row
/// holds `2 * cstride` values -- one side's worth of data per
/// coefficient immediately followed by the other's -- and
/// `num_int_entities` is twice the number of facets; see the packing
/// code below for the exact per-facet layout.
template <dolfinx::scalar T, std::floating_point U>
void pack_coefficients(const Form<T, U>& form,
                       std::map<std::pair<IntegralType, int>,
                                std::pair<std::vector<T>, int>>& coeffs)
{
  const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients
      = form.coefficients();
  const std::vector<int> offsets = form.coefficient_offsets();

  for (auto& [integral_key, coeff_data] : coeffs)
  {
    auto [integral_type, idx] = integral_key;
    std::vector<T>& c = coeff_data.first;
    int cstride = coeff_data.second;
    if (!coefficients.empty())
    {
      switch (integral_type)
      {
      case IntegralType::cell:
      {
        // `form.mesh()` is fixed for the whole call, so its dimension
        // is fetched once rather than once per active coefficient.
        const int form_tdim = form.mesh()->topology()->dim();

        // Iterate over coefficients that are active in cell integrals
        for (int coeff : form.active_coeffs(IntegralType::cell, idx))
        {
          // Get coefficient mesh
          auto mesh = coefficients[coeff]->function_space()->mesh();
          assert(mesh);

          // A cell-integral coefficient must be defined over cells (or
          // a mesh view of them), not lower-codimension entities such
          // as facets -- that combination doesn't make sense and is a
          // logic error, so fail loudly rather than pack it anyway.
          if (int codim = form_tdim - mesh->topology()->dim(); codim > 0)
          {
            throw std::runtime_error("Should not be packing coefficients with "
                                     "codim>0 in a cell integral");
          }

          std::span<const std::int32_t> cells_b
              = form.domain_coeff(IntegralType::cell, idx, coeff);
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
        for (int coeff : form.active_coeffs(IntegralType::interior_facet, idx))
        {
          auto mesh = coefficients[coeff]->function_space()->mesh();
          std::span<const std::int32_t> facets_b
              = form.domain_coeff(IntegralType::interior_facet, idx, coeff);
          md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 4>>
              facets(facets_b.data(), facets_b.size() / 4, 4);

          std::span<const std::uint32_t> cell_info
              = impl::get_cell_orientation_info(*coefficients[coeff]);

          // Data for the '+' and '-' sides of coefficient `coeff` are
          // interleaved per-coefficient (not stored as two contiguous
          // blocks), i.e. layout is [coeff0 '+', coeff0 '-', coeff1
          // '+', coeff1 '-', ...]. `2 * offsets[coeff]` is therefore
          // the start of coefficient `coeff`'s '+' data, immediately
          // followed by its '-' data at `offsets[coeff] +
          // offsets[coeff + 1]`.

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
        // Iterate over coefficients that are active in exterior_facet,
        // vertex, and ridge integrals (all use the same (cell,
        // local_entity_index) entity layout)
        for (int coeff : form.active_coeffs(integral_type, idx))
        {
          // Get coefficient mesh
          auto mesh = coefficients[coeff]->function_space()->mesh();
          assert(mesh);

          std::span<const std::int32_t> entities_b
              = form.domain_coeff(integral_type, idx, coeff);
          md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 2>>
              entities(entities_b.data(), entities_b.size() / 2, 2);
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
/// @tparam T Scalar type of the coefficient.
/// @tparam U Floating point type of the mesh geometry.
/// @param[in] coeff The coefficient to extract cell indices for.
/// @param[in] mesh The mesh which the integration entities belong to.
/// @param[in] entities The integration entities. Is either a sequence of local
/// cell indices, or a sequence of (cell, local entity index) tuples.
/// @param[in] entity_map The map between `mesh` and `coeff`'s mesh.
/// Required (must have a value) whenever `coeff` is not defined on
/// `mesh` itself.
/// @return A vector of cell indices on the coefficient mesh corresponding to
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
          contiguous_cells.push_back(
              c_to_e->links(entities(e, 0))[entities(e, 1)]);
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
/// @param entities Entities to pack over: either a rank-1 list of cell
/// indices, or a rank-2 list of (cell, local_entity_index) pairs.
/// @param entity_maps Bidirectional maps between the entities of a
/// parent mesh and a submesh in case of coefficients being defined on
/// both.
/// @param offsets Insertion offset for each of the `coeffs` when packed
/// into `c`.
/// @param[in,out] c Packed coefficients.
template <dolfinx::scalar T, std::floating_point U>
void pack_coefficients(
    const std::vector<std::reference_wrapper<const Function<T, U>>>& coeffs,
    const mesh::Mesh<U>& mesh, fem::MDSpan2 auto entities,
    const std::vector<std::reference_wrapper<const dolfinx::mesh::EntityMap>>&
        entity_maps,
    std::span<const int> offsets, std::span<T> c)
{

  assert(!offsets.empty());
  const int cstride = offsets.back();

  if (c.size() < entities.extent(0) * offsets.back())
    throw std::runtime_error("Coefficient packing span is too small.");

  // Helper function to get correct entity map. Note: `mesh` is
  // captured by reference -- capturing it by value would copy the
  // whole Mesh (including its Geometry's coordinate array) on every
  // call.
  auto get_entity_map
      = [&mesh, &entity_maps](auto& mesh0) -> const mesh::EntityMap&
  {
    auto it = std::ranges::find_if(
        entity_maps,
        [&mesh, mesh0](const mesh::EntityMap& em)
        {
          return (em.topology() == mesh0->topology()
                  and em.sub_topology() == mesh.topology())
                 or (em.sub_topology() == mesh0->topology()
                     and em.topology() == mesh.topology());
        });

    if (it == entity_maps.end())
    {
      throw std::runtime_error(
          "Incompatible mesh. argument entity_maps must be provided.");
    }
    return *it;
  };

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
/// @tparam T Scalar type of the constants.
/// @param c Constants to pack.
/// @return Packed constants, as the concatenation of each constant's
/// values in order.
template <typename T>
std::vector<T> pack_constants(
    const std::vector<std::reference_wrapper<const fem::Constant<T>>>& c)
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
  c.reserve(u.constants().size());
  std::ranges::transform(u.constants(), std::back_inserter(c),
                         [](auto& c) -> const Constant<T>& { return *c; });
  return fem::pack_constants(c);
}

} // namespace dolfinx::fem
