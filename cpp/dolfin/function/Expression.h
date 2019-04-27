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
  /// @param value_shape (std::vector<std::size_t>)
  ///         Shape of expression.
  explicit Expression(std::vector<std::size_t> value_shape);

  /// Create tensor-valued expression with given shape.
  ///
  /// @param value_shape (std::vector<std::size_t>)
  ///         Shape of expression.
  Expression(std::function<void(PetscScalar*, const double*, const int64_t*,
                                int, int, int, int, double)>
                 eval_ptr,
             std::vector<std::size_t> value_shape);

  /// Copy constructor
  ///
  /// @param expression (Expression)
  ///         Object to be copied.
  Expression(const Expression& expression) = default;

  /// Destructor
  virtual ~Expression() = default;

  /// Return value rank.
  ///
  /// @return std::size_t
  ///         The value rank.
  virtual std::size_t value_rank() const;

  /// Return value dimension for given axis.
  ///
  /// @param i (std::size_t)
  ///         Integer denoting the axis to use.
  ///
  /// @return std::size_t
  ///         The value dimension (for the given axis).
  virtual std::size_t value_dimension(std::size_t i) const;

  /// Return value shape
  ///
  /// @return std::vector<std::size_t>
  ///         The value shape.
  virtual std::vector<std::size_t> value_shape() const;

  /// Restrict function to local cell (compute expansion coefficients w).
  ///
  /// @param    w (list of PetscScalar)
  ///         Expansion coefficients.
  /// @param    element (_FiniteElement_)
  ///         The element.
  /// @param    dolfin_cell (_Cell_)
  ///         The cell.
  /// @param  coordinate_dofs (double*)
  ///         The coordinates
  virtual void
  restrict(PetscScalar* w, const fem::FiniteElement& element,
           const mesh::Cell& dolfin_cell,
           const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs) const;

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
  /// @param    cell (mesh::Cell)
  ///         The cell which contains the given point.
  virtual void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor>>
                        values,
                    const Eigen::Ref<const EigenRowArrayXXd> x,
                    const dolfin::mesh::Cell& cell) const;

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
  // @param x
  //        Pointer to a row major C-style 2D array of `double`.
  //        The array has shape=(number of points, geometrical dimension)
  //        and represents array of points in physical space at which the
  //        Expression is being evaluated.
  // @param cell_idx
  //        Pointer to a 1D C-style array of `int`. It is an array
  //        of indices of cells where points are evaluated. Value -1 represents
  //        cell-independent eval function
  // @param num_points
  //        Number of points where expression is evaluated
  // @param value_size
  //        Size of expression value
  // @param gdim
  //        Geometrical dimension of physical point where expression
  //        is evaluated
  // @param num_cells
  //        Number of cells
  // @param t
  //        Time
  std::function<void(PetscScalar* values, const double* x,
                     const int64_t* cell_idx, int num_points, int value_size,
                     int gdim, int num_cells, double t)>
      _eval_ptr;

  // Value shape
  std::vector<std::size_t> _value_shape;
};
} // namespace function
} // namespace dolfin
