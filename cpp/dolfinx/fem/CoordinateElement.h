// Copyright (C) 2018-2020 Garth N. Wells and Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "ElementDofLayout.h"
#include <basix/element-families.h>
#include <cstdint>
#include <dolfinx/common/math.h>
#include <dolfinx/mesh/cell_types.h>
#include <functional>
#include <memory>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

namespace basix
{
class FiniteElement;
}

namespace dolfinx::fem
{

// FIXME: A dof layout on a reference cell needs to be defined.
/// This class manages coordinate mappings for isoparametric cells.

class CoordinateElement
{
public:
  /// Create a coordinate element from a Basix element
  /// @param[in] element Element from Basix
  explicit CoordinateElement(std::shared_ptr<basix::FiniteElement> element);

  /// Create a Lagrage coordinate element
  /// @param[in] celltype The cell shape
  /// @param[in] degree Polynomial degree of the map
  /// @param[in] type The type of Lagrange element (see Basix
  /// documentation for possible types)
  CoordinateElement(mesh::CellType celltype, int degree,
                    basix::element::lagrange_variant type
                    = basix::element::lagrange_variant::equispaced);

  /// Destructor
  virtual ~CoordinateElement() = default;

  /// Cell shape
  /// @return The cell shape
  mesh::CellType cell_shape() const;

  /// Return the topological dimension of the cell shape
  int topological_dimension() const;

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
  /// The shape of x is (number of points, geometric dimension).
  /// @return The basis functions (and derivatives). The shape is
  /// (derivative, number point, number of basis fn, value size).
  xt::xtensor<double, 4> tabulate(int nd,
                                  const xt::xtensor<double, 2>& X) const;

  /// Evaluate basis values and derivatives at set of points.
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only.
  /// @param[in] X The points at which to compute the basis functions.
  /// The shape of X is (number of points, geometric dimension).
  /// @param[out] basis The array to fill with the basis function
  /// values. The shape can be computed using
  /// `FiniteElement::tabulate_shape`
  void tabulate(int nd, const xt::xtensor<double, 2>& X,
                xt::xtensor<double, 4>& basis) const;

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
    const int gdim = J.shape(1);
    const int tdim = K.shape(1);
    if (gdim == tdim)
      math::inv(J, K);
    else
      math::pinv(J, K);
  }

  /// Compute the determinant of the Jacobian
  /// @param[in] J Jacobian (shape=(gdim, tdim))
  /// @return Determinant of `J`
  template <typename U>
  static double compute_jacobian_determinant(const U& J)
  {
    if (J.shape(0) == J.shape(1))
      return math::det(J);
    else
    {
      using T = typename U::value_type;
      auto B = xt::transpose(J);
      xt::xtensor<T, 2> BA = xt::zeros<T>({B.shape(0), J.shape(1)});
      math::dot(B, J, BA);
      return std::sqrt(math::det(BA));
    }
  }

  /// Compute and return the dof layout
  ElementDofLayout dof_layout() const;

  /// Compute physical coordinates x for points X  in the reference
  /// configuration
  /// @param[in,out] x The physical coordinates of the reference points X
  /// @param[in] cell_geometry The cell node coordinates (physical)
  /// @param[in] phi Tabulated basis functions at reference points X
  static void push_forward(xt::xtensor<double, 2>& x,
                           const xt::xtensor<double, 2>& cell_geometry,
                           const xt::xtensor<double, 2>& phi);

  /// Compute the physical coordinate of the reference point X=(0 , 0,
  /// 0)
  /// @param[in] cell_geometry The cell nodes coordinates (shape=(num
  /// geometry nodes, gdim))
  /// @return Physical coordinate of the X=(0, 0, 0)
  /// @note Assumes that x0 is given by cell_geometry(0, i)
  static std::array<double, 3> x0(const xt::xtensor<double, 2>& cell_geometry);

  /// Compute reference coordinates X for physical coordinates x for an
  /// affine map. For the affine case, `x = J X + x0`, and this function
  /// computes `X = K(x -x0)` where `K = J^{-1}`.
  /// @param[out] X The reference coordinates to compute
  /// (shape=(num_points, tdim))
  /// @param[in] K The inverse of the geometry Jacobian (shape=(tdim,
  /// gdim))
  /// @param[in] x0 The physical coordinate of reference coordinate X0=(0, 0, 0).
  /// @param[in] x The physical coordinates (shape=(num_points, gdim))
  static void pull_back_affine(xt::xtensor<double, 2>& X,
                               const xt::xtensor<double, 2>& K,
                               const std::array<double, 3>& x0,
                               const xt::xtensor<double, 2>& x);

  /// Compute reference coordinates X for physical coordinates x for a
  /// non-affine map.
  /// @param [in,out] X The reference coordinates to compute (shape=(num_points, tdim))
  /// @param [in] x The physical coordinates (shape=(num_points, gdim))
  /// @param [in] cell_geometry The cell nodes coordinates (shape=(num
  /// geometry nodes, gdim))
  /// @param [in] tol Tolerance for termination of Newton method.
  /// @param [in] maxit Maximum number of Newton iterations
  /// @note If convergence is not achieved within maxit, the function throws a run-time error.
  void pull_back_nonaffine(xt::xtensor<double, 2>& X,
                           const xt::xtensor<double, 2>& x,
                           const xt::xtensor<double, 2>& cell_geometry,
                           double tol = 1.0e-8, int maxit = 10) const;

  /// Permutes a list of DOF numbers on a cell
  void permute_dofs(const xtl::span<std::int32_t>& dofs,
                    std::uint32_t cell_perm) const;

  /// Reverses a DOF permutation
  void unpermute_dofs(const xtl::span<std::int32_t>& dofs,
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
  std::shared_ptr<basix::FiniteElement> _element;
};
} // namespace dolfinx::fem
