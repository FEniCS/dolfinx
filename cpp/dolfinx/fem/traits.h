// Copyright (C) 2024-2026 Joseph P. Dean and Garth N. Wells
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/mdspan.hpp>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <span>
#include <tuple>
#include <type_traits>

namespace dolfinx::fem
{
/// @brief DOF transform kernel concept.
template <class U, class T>
concept DofTransformKernel
    = std::is_invocable_v<U, std::span<T>, std::span<const std::uint32_t>,
                          std::int32_t, int>;

/// @brief Whether a DofTransformKernel `fn` should be invoked.
///
/// A nullable kernel (`std::function`) is checked for truthiness,
/// matching the "no transform needed" convention used throughout the
/// assembly/interpolation code. A non-nullable callable (e.g. a plain
/// lambda, as used when calling the low-level `impl::assemble_*`
/// kernels directly -- see the `custom_kernel` demo) can never be
/// "unset", so it is always invoked.
template <typename F>
constexpr bool is_transform_set(const F& fn)
{
  if constexpr (requires { static_cast<bool>(fn); })
    return static_cast<bool>(fn);
  else
    return true;
}

/// @brief Finite element cell kernel concept.
///
/// Kernel functions that can be passed to an assembler for execution
/// must satisfy this concept.
template <class U, class T, class G = dolfinx::scalar_value_t<T>>
concept FEkernel = std::is_invocable_v<U, T*, const T*, const T*, const G*,
                                       const int*, const std::uint8_t*, void*>;

/// @brief Concept for mdspan of rank 1 or 2.
template <class T>
concept MDSpan2
    = std::is_convertible_v<
          std::remove_cvref_t<T>,
          md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>>
      or std::is_convertible_v<
          std::remove_cvref_t<T>,
          md::mdspan<const std::int32_t, md::dextents<std::size_t, 1>>>;

/// @brief Concept for a rank-2 mdspan of 32-bit indices.
///
/// The extents may be static or dynamic.
template <class T>
concept MDSpan2Int32
    = dolfinx::MDSpanRank2<T>
      and std::same_as<typename std::remove_cvref_t<T>::value_type,
                       std::int32_t>;

/// @brief Concept for a rank-2 mdspan of a floating-point type.
///
/// The extents may be static or dynamic.
template <class T, class U>
concept MDSpan2Floating
    = std::floating_point<U> and dolfinx::MDSpanRank2<T>
      and std::same_as<typename std::remove_cvref_t<T>::value_type, U>;

/// @cond
/// Common part of the `DofMapPack*` concepts: a 3-tuple whose (0)
/// entry is the dofmap (a rank-2 `const std::int32_t` mdspan) and (1)
/// entry is the block size, as a run-time `int` or a compile-time
/// `std::integral_constant<int, N>`. The (2) entry (cell/entity
/// indices) is constrained separately by each `DofMapPack*` concept,
/// since its shape differs between the cell, entity and facet
/// assembly kernels.
template <class T>
concept DofMapPackBase = requires(const std::remove_cvref_t<T>& t) {
  requires std::tuple_size_v<std::remove_cvref_t<T>> == 3;
  requires MDSpan2Int32<decltype(std::get<0>(t))>;
  { std::get<1>(t) } -> std::convertible_to<int>;
};
/// @endcond

/// @brief Concept for the degree-of-freedom map data passed to the
/// cell assembly kernel, whose (2) entry is a flat, integer-indexable
/// list of cell indices.
template <class T>
concept DofMapPackCells
    = DofMapPackBase<T> and requires(const std::remove_cvref_t<T>& t) {
        { std::get<2>(t)[0] } -> std::convertible_to<std::int32_t>;
      };

/// @brief Concept for the degree-of-freedom map data passed to the
/// entity assembly kernel, whose (2) entry is indexed by (entity,
/// local index).
template <class T>
concept DofMapPackEntities
    = DofMapPackBase<T> and requires(const std::remove_cvref_t<T>& t) {
        { std::get<2>(t)(0, 0) } -> std::convertible_to<std::int32_t>;
      };

/// @brief Concept for the degree-of-freedom map data passed to the
/// interior facet assembly kernel, whose (2) entry is indexed by
/// (facet, side, local index).
template <class T>
concept DofMapPackFacets
    = DofMapPackBase<T> and requires(const std::remove_cvref_t<T>& t) {
        { std::get<2>(t)(0, 0, 0) } -> std::convertible_to<std::int32_t>;
      };
} // namespace dolfinx::fem
