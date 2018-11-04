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

  /// Copy constructor
  ///
  /// @param expression (Expression)
  ///         Object to be copied.
  Expression(const Expression& expression);

  /// Destructor
  virtual ~Expression();

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

  /// Evaluate at given point.
  ///
  /// @param values (Eigen::Ref<Eigen::VectorXd>)
  ///         The values at the point.
  /// @param x (Eigen::Ref<const Eigen::VectorXd>)
  ///         The coordinates of the point.
  virtual void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor>>
                        values,
                    const Eigen::Ref<const EigenRowArrayXXd> x) const;

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

  /// Property setter for type "double"
  /// Used in pybind11 Python interface to attach a value to a python attribute
  ///
  virtual void set_property(std::string name, PetscScalar value);

  /// Property getter for type "double"
  /// Used in pybind11 Python interface to get the value of a python attribute
  ///
  virtual PetscScalar get_property(std::string name) const;

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

private:
  // Value shape
  std::vector<std::size_t> _value_shape;
};
} // namespace function
} // namespace dolfin
