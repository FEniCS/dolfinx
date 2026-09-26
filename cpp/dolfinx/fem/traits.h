// Copyright (C) 2024-2026 Joseph P. Dean and Garth N. Wells
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/mdspan.hpp>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <ranges>
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
/// assembly/interpolation code. A non-nullable callable (a lambda, or
/// a function passed directly, as when calling the low-level
/// `impl::assemble_*` kernels -- see the `custom_kernel` demo) can
/// never be "unset", so it is always invoked.
template <typename F>
constexpr bool is_transform_set(const F& fn)
{
  // A reference to a function is never null, and converting one to
  // bool warns under -Waddress rather than answering the question.
  if constexpr (std::is_function_v<F>)
    return true;
  else if constexpr (requires { static_cast<bool>(fn); })
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

/// @brief Mesh geometry data passed to the assembly kernels.
///
/// @tparam D Geometry dofmap type, a rank-2 mdspan of
/// `const std::int32_t`.
/// @tparam U Geometry (coordinate) scalar type.
template <class D, std::floating_point U>
  requires MDSpan2Int32<D>
struct GeometryPack
{
  /// Geometry dofmap, shape `(num_cells, num_nodes_per_cell)`.
  D dofmap;

  /// Node coordinates, shape `(num_nodes, 3)`. The trailing extent is
  /// static so the coordinate gather folds.
  md::mdspan<const U, md::extents<std::size_t, md::dynamic_extent, 3>> x;
};

/// @brief Degree-of-freedom map data for one form argument, as passed
/// to the assembly kernels.
///
/// A member rather than a tuple element, so that reading the block
/// size cannot introduce a reference. A structured binding of a
/// tuple-like type binds references, and a reference is not usable in
/// a constant expression, silently costing the compile-time block
/// size.
///
/// @tparam D Dofmap type, a rank-2 mdspan of `const std::int32_t`.
/// @tparam B Block size type, `int` or
/// `std::integral_constant<int, N>`.
/// @tparam E Entity index list type, constrained by the
/// `DofMapPack*` concepts below.
template <class D, class B, class E>
struct DofMapPack
{
  /// Dofmap, shape `(num_cells, num_dofs_per_cell)`.
  D map;

  /// Dofmap block size.
  B bs;

  /// Entity indices in this argument's mesh.
  E entities;
};

/// @cond
/// Common part of the `DofMapPack*` concepts.
template <class T>
concept DofMapPackBase = requires(const std::remove_cvref_t<T>& t) {
  requires MDSpan2Int32<decltype(t.map)>;
  { t.bs } -> std::convertible_to<int>;
};
/// @endcond

/// @brief Concept for the degree-of-freedom map data passed to the
/// cell assembly kernel, whose entities are a flat, integer-indexable
/// list of cell indices.
template <class T>
concept DofMapPackCells
    = DofMapPackBase<T> and requires(const std::remove_cvref_t<T>& t) {
        { t.entities[0] } -> std::convertible_to<std::int32_t>;
      };

/// @brief Concept for the degree-of-freedom map data passed to the
/// entity assembly kernel, whose entities are indexed by (entity,
/// local index).
template <class T>
concept DofMapPackEntities
    = DofMapPackBase<T> and requires(const std::remove_cvref_t<T>& t) {
        { t.entities(0, 0) } -> std::convertible_to<std::int32_t>;
      };

/// @brief Concept for the degree-of-freedom map data passed to the
/// interior facet assembly kernel, whose entities are indexed by
/// (facet, side, local index).
template <class T>
concept DofMapPackFacets
    = DofMapPackBase<T> and requires(const std::remove_cvref_t<T>& t) {
        { t.entities(0, 0, 0) } -> std::convertible_to<std::int32_t>;
      };

/// @brief Data for one form argument (test or trial function) passed
/// to the assembly kernels.
///
/// @tparam P Dof transformation kernel type.
/// @tparam D Dofmap type.
/// @tparam B Block size type.
/// @tparam E Entity index list type.
template <class P, class D, class B, class E>
struct FormArgument
{
  /// Dofmap, block size and entity indices for this argument.
  DofMapPack<D, B, E> dofmap;

  /// Dof transformation applied in-place to the element tensor. Held
  /// by reference: it is a `std::function`, and the kernels must not
  /// allocate.
  const P& transform;

  /// Cell permutations for this argument's mesh. Empty if the element
  /// needs no dof transformations.
  std::span<const std::uint32_t> cell_info;
};

/// @cond
/// Common part of the `FormArgument*` concepts.
template <class A, class T>
concept FormArgumentBase = requires(const std::remove_cvref_t<A>& a) {
  requires DofTransformKernel<std::remove_cvref_t<decltype(a.transform)>, T>;
  { a.cell_info } -> std::convertible_to<std::span<const std::uint32_t>>;
};
/// @endcond

/// @brief Concept for the form argument data passed to the cell
/// assembly kernels.
template <class A, class T>
concept FormArgumentCells
    = FormArgumentBase<A, T> and requires(const std::remove_cvref_t<A>& a) {
        requires DofMapPackCells<decltype(a.dofmap)>;
      };

/// @brief Concept for the form argument data passed to the entity
/// assembly kernels.
template <class A, class T>
concept FormArgumentEntities
    = FormArgumentBase<A, T> and requires(const std::remove_cvref_t<A>& a) {
        requires DofMapPackEntities<decltype(a.dofmap)>;
      };

/// @brief Concept for the form argument data passed to the interior
/// facet assembly kernels.
template <class A, class T>
concept FormArgumentFacets
    = FormArgumentBase<A, T> and requires(const std::remove_cvref_t<A>& a) {
        requires DofMapPackFacets<decltype(a.dofmap)>;
      };

/// @brief Concept for a randomly-indexable list of process-local
/// indices, as used for the cell lists passed to the assembly kernels.
///
/// Satisfied by `std::span<const std::int32_t>`, by a span or array of
/// static extent, which carries the list length in its type, and by a
/// generated range such as `std::views::iota`. A generated range costs
/// no memory traffic: the assembler's cell lookup folds to the loop
/// index, which is what a caller assembling over every cell wants.
template <class C>
concept IndexList
    = std::ranges::random_access_range<C>
      and std::same_as<std::ranges::range_value_t<C>, std::int32_t>;

/// @brief Concept for the mutable scratch buffers passed to the
/// assembly kernels.
///
/// Satisfied by `std::span<T>` and by `std::array<T, N>`. The buffers
/// are taken by value, so an array carries its size in its type and the
/// assembler can size its work at compile time; a span leaves the size
/// to run time and the storage to the caller.
template <class B, class T>
concept ScratchBuffer
    = std::ranges::contiguous_range<B> and std::ranges::output_range<B, T>
      and std::same_as<std::ranges::range_value_t<B>, T> and requires(B& b) {
            { b.data() } -> std::same_as<T*>;
            { b.size() } -> std::convertible_to<std::size_t>;
          };

/// @brief Concept for the container that assembled values are
/// accumulated into, indexed by a process-local degree-of-freedom
/// index.
template <class V, class T>
concept AssemblyVector
    = std::same_as<typename std::remove_cvref_t<V>::value_type, T>
      and requires(std::remove_cvref_t<V>& v, std::int32_t i) {
            { v[i] } -> std::convertible_to<T&>;
          };

namespace impl
{
/// @brief Rank-2 mdspan of 32-bit indices, as used for the dofmaps
/// passed to the assembly kernels.
using mdspan2_t = md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>;

/// @brief Call `f` with the dofmap block size as a compile-time
/// constant for the common block sizes, and as a plain `int`
/// otherwise.
///
/// The kernels loop over the block size when scattering the element
/// tensor, so `std::integral_constant<int, N>` lets that loop unroll
/// and the offsets fold. Other block sizes take the run-time path.
///
/// @param[in] bs Dofmap block size.
/// @param[in] f Callable invoked with the block size.
template <class F>
void dispatch_bs(int bs, F&& f)
{
  switch (bs)
  {
  case 1:
    return f(std::integral_constant<int, 1>{});
  case 3:
    return f(std::integral_constant<int, 3>{});
  default:
    return f(bs);
  }
}

/// @brief Call `f` with the test and trial function block sizes as
/// compile-time constants when they are equal and one of the common
/// block sizes, and as plain `int`s otherwise.
///
/// See the single block size overload. Only matching sizes are
/// specialised; the mixed cases would multiply instantiations for a
/// case a bilinear form rarely has.
///
/// @param[in] bs0 Test function dofmap block size.
/// @param[in] bs1 Trial function dofmap block size.
/// @param[in] f Callable invoked with the two block sizes.
template <class F>
void dispatch_bs(int bs0, int bs1, F&& f)
{
  if (bs0 == bs1)
  {
    switch (bs0)
    {
    case 1:
      return f(std::integral_constant<int, 1>{},
               std::integral_constant<int, 1>{});
    case 3:
      return f(std::integral_constant<int, 3>{},
               std::integral_constant<int, 3>{});
    }
  }

  return f(bs0, bs1);
}
} // namespace impl
} // namespace dolfinx::fem
