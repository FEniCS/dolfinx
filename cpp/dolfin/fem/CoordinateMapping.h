// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/types.h>
#include <memory>
#include <ufc.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace dolfin
{

namespace fem
{

/// This is a wrapper for a UFC coordinate mapping (ufc::coordinate_mapping).

class CoordinateMapping
{
public:
  /// Create coordinate mapping from UFC coordinate mappping (data may be
  /// shared)
  /// @param cm (ufc::coordinate_mapping)
  ///  UFC coordinate mapping
  CoordinateMapping(std::shared_ptr<const ufc::coordinate_mapping> cm)
      : _ufc_cm(cm)
  {
    // Do nothing
  }

  /// Destructor
  virtual ~CoordinateMapping() {}

  //--- Direct wrappers for ufc::coordinate_mapping ---

  /// Return a string identifying the finite element
  /// @return std::string
  std::string signature() const
  {
    assert(_ufc_cm);
    return _ufc_cm->signature();
  }

  /// Return the cell shape
  /// @return ufc::shape
  ufc::shape cell_shape() const
  {
    assert(_ufc_cm);
    return _ufc_cm->cell_shape();
  }

  /// Return the topological dimension of the cell shape
  /// @return std::size_t
  std::uint32_t topological_dimension() const
  {
    assert(_ufc_cm);
    return _ufc_cm->topological_dimension();
  }

  /// Return the geometric dimension of the cell shape
  /// @return std::uint32_t
  std::uint32_t geometric_dimension() const
  {
    assert(_ufc_cm);
    return _ufc_cm->geometric_dimension();
  }

  /// Compute physical coordinates x for points X  in the reference
  /// configuration
  void compute_physical_coordinates(
      Eigen::Ref<EigenRowArrayXXd> x,
      const Eigen::Ref<const EigenRowArrayXXd>& X,
      const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs) const
  {
    assert(_ufc_cm);
    assert(x.rows() == X.rows());
    assert(x.cols() == _ufc_cm->geometric_dimension());
    assert(X.cols() == _ufc_cm->topological_dimension());
    _ufc_cm->compute_physical_coordinates(x.data(), X.rows(), X.data(),
                                          coordinate_dofs.data());
  }

  /// Compute reference coordinates X, and J, detJ and K for physical
  /// coordinates x
  void compute_reference_geometry(
      EigenRowArrayXXd& X, Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
      EigenArrayXd& detJ, Eigen::Tensor<double, 3, Eigen::RowMajor>& K,
      const Eigen::Ref<const EigenRowArrayXXd>& x,
      const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs) const
  {
    // Number of points
    int num_points = x.rows();

    // In checks
    assert(x.cols() == this->geometric_dimension());
    // assert(coordinate_dofs.rows() == space_dimension);
    assert(coordinate_dofs.cols() == this->geometric_dimension());

    // In/out size checks
    assert(X.rows() == num_points);
    assert(X.cols() == this->topological_dimension());
    assert(J.dimension(0) == num_points);
    assert(J.dimension(1) == this->geometric_dimension());
    assert(J.dimension(2) == this->topological_dimension());
    assert(detJ.rows() == num_points);
    assert(K.dimension(0) == num_points);
    assert(K.dimension(1) == this->topological_dimension());
    assert(K.dimension(2) == this->geometric_dimension());

    assert(_ufc_cm);
    _ufc_cm->compute_reference_geometry(X.data(), J.data(), detJ.data(),
                                        K.data(), num_points, x.data(),
                                        coordinate_dofs.data(), 1);
  }

  // /// Return underlying UFC coordinate_mapping. Intended for libray usage
  // only
  // /// and may change.
  // std::shared_ptr<const ufc::finite_element> ufc_element() const
  // {
  //   return _ufc_element;
  // }

private:
  // UFC finite element
  std::shared_ptr<const ufc::coordinate_mapping> _ufc_cm;
};
}
}
