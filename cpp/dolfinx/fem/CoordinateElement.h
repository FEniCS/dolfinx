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
#include <cstdint>
#include <dolfinx/common/math.h>
#include <dolfinx/mesh/cell_types.h>
#include <memory>
#include <span>

namespace basix
{
class FiniteElement;
}

namespace dolfinx::fem
{

/// A CoordinateElement manages coordinate mappings for isoparametric
/// cells.
/// @todo A dof layout on a reference cell needs to be defined.
class CoordinateElement
{
public:
  /// Create a coordinate element from a Basix element
  /// @param[in] element Element from Basix
  explicit CoordinateElement(
      std::shared_ptr<const basix::FiniteElement> element);

  /// Create a Lagrange coordinate element
  /// @param[in] celltype The cell shape
  /// @param[in] degree Polynomial degree of the map
  /// @param[in] type The type of Lagrange element (see Basix
  /// documentation for possible types)
  CoordinateElement(mesh::CellType celltype, int degree,
                    basix::element::lagrange_variant type
                    = basix::element::lagrange_variant::unset);

  /// Destructor
  virtual ~CoordinateElement() = default;

  /// Cell shape
  /// @return The cell shape
  mesh::CellType cell_shape() const;

  /// The polynomial degree of the element
  int degree() const;

  /// @brief The dimension of the geometry element space.
  ///
  /// The number of basis function is returned. E.g., for a linear
  /// triangle cell the dimension will be 3.
  ///
  /// @return The coordinate element dimension.
  int dim() const;

  /// The variant of the element
  basix::element::lagrange_variant variant() const;

  /// Shape of array to fill when calling `FiniteElement::tabulate`
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only
  /// @param[in] num_points Number of points at which to evaluate the
  /// basis functions
  /// @return The shape of the array to be filled by `FiniteElement::tabulate`
  std::array<std::size_t, 4> tabulate_shape(std::size_t nd,
                                            std::size_t num_points) const;

  /// Evaluate basis values and derivatives at set of points.
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only.
  /// @param[in] X The points at which to compute the basis functions.
  /// The shape of X is (number of points, geometric dimension).
  /// @param[in] shape The shape of `X`.
  /// @param[out] basis The array to fill with the basis function
  /// values. The shape can be computed using
  /// `FiniteElement::tabulate_shape`
  void tabulate(int nd, std::span<const double> X,
                std::array<std::size_t, 2> shape,
                std::span<double> basis) const;

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

  /// Compute the inverse of the Jacobian
  /// @param[in] J The Jacobian (shape=(gdim, tdim))
  /// @param[out] K The Jacobian (shape=(tdim, gdim))
  template <typename U, typename V>
  static void compute_jacobian_inverse(const U& J, V&& K)
  {
    const int gdim = J.extent(0);
    const int tdim = K.extent(0);
    if (gdim == tdim)
      math::inv(J, K);
    else
      math::pinv(J, K);
  }

  /// Compute the determinant of the Jacobian
  /// @param[in] J Jacobian (shape=(gdim, tdim))
  /// @param[in] w Working memory, required when gdim != tdim. Size
  /// must be at least 2 * gdim * tdim.
  /// @return Determinant of `J`
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

      using T = typename U::element_type;
      namespace stdex = std::experimental;
      using mdspan2_t = stdex::mdspan<T, stdex::dextents<std::size_t, 2>>;
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
  /// reference configuration
  /// @param[in,out] x The physical coordinates of the reference points
  /// X (rank 2)
  /// @param[in] cell_geometry The cell node physical coordinates (rank 2)
  /// @param[in] phi Tabulated basis functions at reference points X (rank 2)
  template <typename U, typename V, typename W>
  static void push_forward(U&& x, const V& cell_geometry, const W& phi)
  {
    for (std::size_t i = 0; i < x.extent(0); ++i)
      for (std::size_t j = 0; j < x.extent(1); ++j)
        x(i, j) = 0;

    // Compute x = phi * cell_geometry;
    math::dot(phi, cell_geometry, x);
  }

  /// Compute reference coordinates X for physical coordinates x for an
  /// affine map. For the affine case, `x = J X + x0`, and this function
  /// computes `X = K(x -x0)` where `K = J^{-1}`.
  /// @param[out] X The reference coordinates to compute
  /// (shape=(num_points, tdim))
  /// @param[in] K The inverse of the geometry Jacobian (shape=(tdim,
  /// gdim))
  /// @param[in] x0 The physical coordinate of reference coordinate X0=(0, 0,
  /// 0).
  /// @param[in] x The physical coordinates (shape=(num_points, gdim))
  template <typename U, typename V, typename W>
  static void pull_back_affine(U&& X, const V& K,
                               const std::array<double, 3>& x0, const W& x)
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
  using mdspan2_t
      = std::experimental::mdspan<double,
                                  std::experimental::dextents<std::size_t, 2>>;
  /// mdspan typedef
  using cmdspan2_t
      = std::experimental::mdspan<const double,
                                  std::experimental::dextents<std::size_t, 2>>;

  /// Compute reference coordinates X for physical coordinates x for a
  /// non-affine map.
  /// @param [in,out] X The reference coordinates to compute
  /// (shape=`(num_points, tdim)`)
  /// @param [in] x The physical coordinates (shape=`(num_points, gdim)`)
  /// @param [in] cell_geometry The cell nodes coordinates (shape=(num
  /// geometry nodes, gdim))
  /// @param [in] tol Tolerance for termination of Newton method.
  /// @param [in] maxit Maximum number of Newton iterations
  /// @note If convergence is not achieved within maxit, the function
  /// throws a runtime error.
  void pull_back_nonaffine(mdspan2_t X, cmdspan2_t x, cmdspan2_t cell_geometry,
                           double tol = 1.0e-8, int maxit = 10) const;

  /// Permutes a list of DOF numbers on a cell
  void permute_dofs(const std::span<std::int32_t>& dofs,
                    std::uint32_t cell_perm) const;

  /// Reverses a DOF permutation
  void unpermute_dofs(const std::span<std::int32_t>& dofs,
                      std::uint32_t cell_perm) const;

  /// Indicates whether the geometry DOF numbers on each cell need
  /// permuting
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
  std::shared_ptr<const basix::FiniteElement> _element;
};
} // namespace dolfinx::fem
