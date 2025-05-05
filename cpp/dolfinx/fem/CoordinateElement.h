// Copyright (C) 2018-2020 Garth N. Wells and Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "ElementDofLayout.h"
#include <algorithm>
#include <array>
#include <basix/element-families.h>
#include <basix/mdspan.hpp>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/math.h>
#include <dolfinx/mesh/cell_types.h>
#include <limits>
#include <memory>
#include <span>

namespace basix
{
template <std::floating_point T>
class FiniteElement;
}

namespace dolfinx::fem
{
/// A CoordinateElement manages coordinate mappings for isoparametric
/// cells.
/// @todo A dof layout on a reference cell needs to be defined.
/// @tparam T Floating point (real) type for the geometry and for the
/// element basis.
template <std::floating_point T>
class CoordinateElement
{
public:
  /// @brief Create a coordinate element from a Basix element.
  /// @param[in] element Basix finite element
  explicit CoordinateElement(
      std::shared_ptr<const basix::FiniteElement<T>> element);

  /// @brief Create a Lagrange coordinate element.
  /// @param[in] celltype Cell shape.
  /// @param[in] degree Polynomial degree of the map.
  /// @param[in] type Type of Lagrange element (see Basix documentation
  /// for possible types).
  CoordinateElement(mesh::CellType celltype, int degree,
                    basix::element::lagrange_variant type
                    = basix::element::lagrange_variant::unset);

  /// Destructor
  virtual ~CoordinateElement() = default;

  /// @brief Cell shape.
  /// @return The cell shape
  mesh::CellType cell_shape() const;

  /// @brief The polynomial degree of the element.
  /// @return The degree
  int degree() const;

  /// @brief The dimension of the coordinate element space.
  ///
  /// The number of basis function is returned. E.g., for a linear
  /// triangle cell the dimension will be 3.
  ///
  /// @return Dimension of the coordinate element space.
  int dim() const;

  /// @brief Variant of the element
  basix::element::lagrange_variant variant() const;

  /// @brief Element hash.
  ///
  /// This is the Basix element hash.
  std::uint64_t hash() const;

  /// @brief Shape of array to fill when calling `tabulate`.
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only
  /// @param[in] num_points Number of points at which to evaluate the
  /// basis functions.
  /// @return Shape of the array to be filled by `tabulate`, where (0)
  /// is derivative index, (1) is the point index, (2) is the basis
  /// function index and (3) is the basis function component.
  std::array<std::size_t, 4> tabulate_shape(std::size_t nd,
                                            std::size_t num_points) const;

  /// @brief Evaluate basis values and derivatives at set of points.
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only.
  /// @param[in] X The points at which to compute the basis functions.
  /// The shape of X is (number of points, geometric dimension).
  /// @param[in] shape The shape of `X`.
  /// @param[out] basis The array to fill with the basis function
  /// values. The shape can be computed using `tabulate_shape`.
  void tabulate(int nd, std::span<const T> X, std::array<std::size_t, 2> shape,
                std::span<T> basis) const;

  /// @brief Given the closure DOFs \f$\tilde{d}\f$ of a cell sub-entity in
  /// reference ordering, this function computes the permuted degrees-of-freedom
  ///   \f[ d = P \tilde{d},\f]
  /// ordered to be consistent with the entity's mesh orientation, where
  /// \f$P\f$ is a permutation matrix. This accounts for orientation
  /// discrepancies between the entity's cell and mesh orientation. All DOFs are
  /// rotated and reflected together, unlike `permute`, which considered
  /// sub-entities independently.
  ///
  /// @param[in,out] d Indices associated with the reference element
  /// degree-of-freedom (in). Indices associated with each physical
  /// element degree-of-freedom (out).
  /// @param[in] cell_info Permutation info for the cell
  /// @param[in] entity_type The cell type of the sub-entity
  /// @param[in] entity_index The local (with respect to the cell) index of the
  /// entity
  void permute_subentity_closure(std::span<std::int32_t> d,
                                 std::uint32_t cell_info,
                                 mesh::CellType entity_type,
                                 int entity_index) const;

  /// Compute Jacobian for a cell with given geometry using the
  /// basis functions and first order derivatives.
  /// @param[in] dphi Derivatives of the basis functions (shape=(tdim,
  /// num geometry nodes))
  /// @param[in] cell_geometry The cell nodes coordinates (shape=(num
  /// geometry nodes, gdim))
  /// @param[out] J The Jacobian. It must have shape=(gdim, tdim) and
  /// must initialized to zero
  template <typename U, typename V, typename W>
  static void compute_jacobian(const U& dphi, const V& cell_geometry, W&& J)
  {
    math::dot(cell_geometry, dphi, J, true);
  }

  /// @brief Compute the inverse of the Jacobian.
  /// @param[in] J Jacobian (shape=(gdim, tdim)).
  /// @param[out] K Inverse Jacobian (shape=(tdim, gdim)).
  template <typename U, typename V>
  static void compute_jacobian_inverse(const U& J, V&& K)
  {
    int gdim = J.extent(0);
    int tdim = K.extent(0);
    if (gdim == tdim)
      math::inv(J, K);
    else
      math::pinv(J, K);
  }

  /// @brief Compute the determinant of the Jacobian.
  /// @param[in] J Jacobian (shape=(gdim, tdim)).
  /// @param[in] w Working memory, required when gdim != tdim. Size
  /// must be at least 2 * gdim * tdim.
  /// @return Determinant of `J`.
  template <typename U>
  static double
  compute_jacobian_determinant(const U& J, std::span<typename U::value_type> w)
  {
    static_assert(U::rank() == 2, "Must be rank 2");
    if (J.extent(0) == J.extent(1))
      return math::det(J);
    else
    {
      assert(w.size() >= 2 * J.extent(0) * J.extent(1));
      using X = typename U::element_type;
      using mdspan2_t = md::mdspan<X, md::dextents<std::size_t, 2>>;
      mdspan2_t B(w.data(), J.extent(1), J.extent(0));
      mdspan2_t BA(w.data() + J.extent(0) * J.extent(1), B.extent(0),
                   J.extent(1));
      for (std::size_t i = 0; i < B.extent(0); ++i)
        for (std::size_t j = 0; j < B.extent(1); ++j)
          B(i, j) = J(j, i);

      // Zero working memory of BA
      std::fill_n(BA.data_handle(), BA.size(), 0);
      math::dot(B, J, BA);
      return std::sqrt(math::det(BA));
    }
  }

  /// Compute and return the dof layout
  ElementDofLayout create_dof_layout() const;

  /// @brief Compute physical coordinates x for points X  in the
  /// reference configuration.
  /// @param[in,out] x The physical coordinates of the reference points
  /// X (rank 2).
  /// @param[in] cell_geometry Cell node physical coordinates (rank 2).
  /// @param[in] phi Tabulated basis functions at reference points X
  /// (rank 2).
  template <typename U, typename V, typename W>
  static void push_forward(U&& x, const V& cell_geometry, const W& phi)
  {
    for (std::size_t i = 0; i < x.extent(0); ++i)
      for (std::size_t j = 0; j < x.extent(1); ++j)
        x(i, j) = 0;

    // Compute x = phi * cell_geometry;
    math::dot(phi, cell_geometry, x);
  }

  /// @brief Compute reference coordinates X for physical coordinates x
  /// for an affine map. For the affine case, `x = J X + x0`, and this
  /// function computes `X = K(x -x0)` where `K = J^{-1}`.
  /// @param[out] X Reference coordinates to compute
  /// (`shape=(num_points, tdim)`),
  /// @param[in] K Inverse of the geometry Jacobian (`shape=(tdim,
  /// gdim)`).
  /// @param[in] x0 Physical coordinate of reference coordinate `X0=(0,
  /// 0, 0)`.
  /// @param[in] x Physical coordinates (shape=(num_points, gdim)).
  template <typename U, typename V, typename W>
  static void pull_back_affine(U&& X, const V& K, const std::array<T, 3>& x0,
                               const W& x)
  {
    assert(X.extent(0) == x.extent(0));
    assert(X.extent(1) == K.extent(0));
    assert(x.extent(1) == K.extent(1));
    for (std::size_t i = 0; i < X.extent(0); ++i)
      for (std::size_t j = 0; j < X.extent(1); ++j)
        X(i, j) = 0;

    // Calculate X for each point
    for (std::size_t p = 0; p < x.extent(0); ++p)
      for (std::size_t i = 0; i < K.extent(0); ++i)
        for (std::size_t j = 0; j < K.extent(1); ++j)
          X(p, i) += K(i, j) * (x(p, j) - x0[j]);
  }

  /// mdspan typedef
  template <typename X>
  using mdspan2_t = md::mdspan<X, md::dextents<std::size_t, 2>>;

  /// @brief Compute reference coordinates `X` for physical coordinates
  /// `x` for a non-affine map.
  /// @param [in,out] X The reference coordinates to compute
  /// (shape=`(num_points, tdim)`).
  /// @param [in] x Physical coordinates (`shape=(num_points, gdim)`).
  /// @param [in] cell_geometry Cell nodes coordinates (`shape=(num
  /// geometry nodes, gdim)`).
  /// @param [in] tol Tolerance for termination of Newton method.
  /// @param [in] maxit Maximum number of Newton iterations
  /// @note If convergence is not achieved within `maxit`, the function
  /// throws a runtime error.
  void pull_back_nonaffine(mdspan2_t<T> X, mdspan2_t<const T> x,
                           mdspan2_t<const T> cell_geometry,
                           double tol = 1.0e-6, int maxit = 15) const;

  /// @brief Permute a list of DOF numbers on a cell.
  void permute(std::span<std::int32_t> dofs, std::uint32_t cell_perm) const;

  /// @brief Reverses a DOF permutation
  void permute_inv(std::span<std::int32_t> dofs, std::uint32_t cell_perm) const;

  /// @brief Indicates whether the geometry DOF numbers on each cell
  /// need permuting.
  ///
  /// For higher order geometries (where there is more than one DOF on a
  /// subentity of the cell), this will be true.
  bool needs_dof_permutations() const;

  /// Check is geometry map is affine
  /// @return True is geometry map is affine
  bool is_affine() const noexcept { return _is_affine; }

private:
  // Flag denoting affine map
  bool _is_affine;

  // Basix Element
  std::shared_ptr<const basix::FiniteElement<T>> _element;
};
} // namespace dolfinx::fem
