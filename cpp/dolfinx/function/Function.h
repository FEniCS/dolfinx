// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/common/types.h>
#include <dolfinx/la/Vector.h>
#include <functional>
#include <memory>
#include <petscsys.h>
#include <petscvec.h>
#include <string>
#include <vector>

namespace dolfinx::function
{

class FunctionSpace;

/// This class represents a function \f$ u_h \f$ in a finite
/// element function space \f$ V_h \f$, given by
///
/// \f[     u_h = \sum_{i=1}^{n} U_i \phi_i \f]
/// where \f$ \{\phi_i\}_{i=1}^{n} \f$ is a basis for \f$ V_h \f$,
/// and \f$ U \f$ is a vector of expansion coefficients for \f$ u_h \f$.

class Function
{
public:
  /// Create function on given function space
  /// @param[in] V The function space
  explicit Function(std::shared_ptr<const FunctionSpace> V);

  /// Create function on given function space with a given vector
  ///
  /// *Warning: This constructor is intended for internal library use only*
  ///
  /// @param[in] V The function space
  /// @param[in] x The vector
  Function(std::shared_ptr<const FunctionSpace> V,
           std::shared_ptr<la::Vector<PetscScalar>> x);

  // Copy constructor
  Function(const Function& v) = delete;

  /// Move constructor
  Function(Function&& v) = default;

  /// Destructor
  virtual ~Function();

  /// Move assignment
  Function& operator=(Function&& v) = default;

  // Assignment
  Function& operator=(const Function& v) = delete;

  /// Extract subfunction (view into the Function)
  /// @param[in] i Index of subfunction
  /// @return The subfunction
  Function sub(int i) const;

  /// Collapse a subfunction (view into the Function) to a stand-alone
  ///   Function
  /// @return New collapsed Function
  Function collapse() const;

  /// Return shared pointer to function space
  /// @return The function space
  std::shared_ptr<const FunctionSpace> function_space() const;

  /// Return vector of expansion coefficients as a PETSc Vec
  /// @return The vector of expansion coefficients
  Vec vector() const;

  /// Underlying vector
  std::shared_ptr<const la::Vector<PetscScalar>> x() const { return _x; }

  /// Underlying vector
  std::shared_ptr<la::Vector<PetscScalar>> x() { return _x; }

  /// Interpolate a Function (on possibly non-matching meshes)
  /// @param[in] v The function to be interpolated.
  void interpolate(const Function& v);

  /// Interpolate an expression
  /// @param[in] f The expression to be interpolated
  void
  interpolate(const std::function<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                               Eigen::Dynamic, Eigen::RowMajor>(
                  const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                                      Eigen::RowMajor>>&)>& f);

  /// Evaluate the Function at points
  ///
  /// @param[in] x The coordinates of the points. It has shape
  ///   (num_points, 3).
  /// @param[in] cells An array of cell indices. cells[i] is the index
  ///   of the cell that contains the point x(i). Negative cell indices
  ///   can be passed, and the corresponding point will be ignored.
  /// @param[in,out] u The values at the points. Values are not computed
  ///   for points with a negative cell index. This argument must be
  ///   passed with the corrext size.
  void
  eval(const Eigen::Ref<
           const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& x,
       const Eigen::Ref<const Eigen::Array<int, Eigen::Dynamic, 1>>& cells,
       Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
           u) const;

  /// Compute values at all mesh 'nodes'
  /// @return The values at all geometric points
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  compute_point_values() const;

  /// Name
  std::string name = "u";

  /// ID
  std::size_t id() const;

private:
  // ID
  std::size_t _id;

  // The function space
  std::shared_ptr<const FunctionSpace> _function_space;

  // The vector of expansion coefficients (local)
  std::shared_ptr<la::Vector<PetscScalar>> _x;

  // PETSc wrapper of the expansion coefficients
  mutable Vec _petsc_vector = nullptr;
};
} // namespace dolfinx::function
