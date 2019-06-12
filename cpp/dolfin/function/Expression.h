// Copyright (C) 2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <memory>
#include <petscsys.h>
#include <vector>

namespace dolfin
{
namespace fem
{
class FiniteElement;
}

namespace mesh
{
class Cell;
class Mesh;
} // namespace mesh

namespace function
{
class FunctionSpace;

class Expression
{

public:
  /// Create tensor-valued expression with given shape.
  ///
  /// @param value_shape (std::vector<int>)
  ///         Shape of expression.
  explicit Expression(std::vector<int> value_shape);

  /// Copy constructor
  ///
  /// @param expression (Expression)
  ///         Object to be copied.
  Expression(const Expression& expression) = default;

  /// Destructor
  virtual ~Expression() = default;

  void set_eval(
      std::function<void(PetscScalar*, int, int, const double*, int, double)>
          eval_ptr);

  /// Return value rank.
  ///
  /// @return int
  ///         The value rank.
  virtual int value_rank() const;

  /// Return value dimension for given axis.
  ///
  /// @param i (int)
  ///         Integer denoting the axis to use.
  ///
  /// @return int
  ///         The value dimension (for the given axis).
  virtual int value_dimension(int i) const;

  /// Return value shape
  ///
  /// @return std::vector<int>
  ///         The value shape.
  virtual std::vector<int> value_shape() const;

  /// Compute values at all mesh vertices.
  ///
  /// @param    mesh (Mesh)
  ///         The mesh.
  /// @returns    vertex_values (EigenRowArrayXXd)
  ///         The values at all vertices.
  virtual Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>
  compute_point_values(const mesh::Mesh& mesh) const;

  /// Evaluate at given point in given cell
  ///
  /// @param    values (Eigen::Ref<Eigen::VectorXd>)
  ///         The values at the point.
  /// @param    x (Eigen::Ref<const Eigen::VectorXd>)
  ///         The coordinates of the point.
  virtual void
  eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
           values,
       const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor>>
           x) const;

  /// Time
  double t = 0.0;

private:
  // Evaluate method
  //
  // Signature of the method accepts:
  // @param values
  //        Pointer to row major 2D C-style array of `PetscScalar`.
  //        The array has shape=(number of points, value size) and has to
  //        be filled with custom values in the function body.
  // @param num_points
  //        Number of points where expression is evaluated
  // @param value_size
  //        Size of expression value
  // @param x
  //        Pointer to a row major C-style 2D array of `double`.
  //        The array has shape=(number of points, geometrical dimension)
  //        and represents array of points in physical space at which the
  //        Expression is being evaluated.
  // @param gdim
  //        Geometrical dimension of physical point where expression
  //        is evaluated
  // @param t
  //        Time
  std::function<void(PetscScalar* values, int num_points, int value_size,
                     const double* x, int gdim, double t)>
      _eval_ptr;

  // Value shape
  std::vector<int> _value_shape;
};
} // namespace function
} // namespace dolfin
