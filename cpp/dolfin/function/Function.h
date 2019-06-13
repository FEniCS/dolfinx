// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/Variable.h>
#include <dolfin/common/types.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/la/PETScVector.h>
#include <memory>
#include <petscsys.h>
#include <petscvec.h>
#include <vector>

namespace dolfin
{

namespace geometry
{
class BoundingBoxTree;
}
namespace mesh
{
class Cell;
class Mesh;
} // namespace mesh

namespace function
{
class Expression;
class FunctionSpace;

/// This class represents a function \f$ u_h \f$ in a finite
/// element function space \f$ V_h \f$, given by
///
/// \f[     u_h = \sum_{i=1}^{n} U_i \phi_i \f]
/// where \f$ \{\phi_i\}_{i=1}^{n} \f$ is a basis for \f$ V_h \f$,
/// and \f$ U \f$ is a vector of expansion coefficients for \f$ u_h \f$.

class Function : public common::Variable
{
public:
  /// Create function on given function space
  ///
  /// @param V (_FunctionSpace_)
  ///         The function space.
  explicit Function(std::shared_ptr<const FunctionSpace> V);

  /// Create function on given function space with a given vector
  ///
  /// *Warning: This constructor is intended for internal library use only*
  ///
  /// @param V (_FunctionSpace_)
  ///         The function space.
  /// @param x (_Vec_)
  ///         The vector.
  Function(std::shared_ptr<const FunctionSpace> V, Vec x);

  // Copy constructor
  Function(const Function& v) = delete;

  /// Move constructor
  Function(Function&& v) = default;

  /// Destructor
  virtual ~Function() = default;

  /// Move assignment
  Function& operator=(Function&& v) = default;

  // Assignment
  Function& operator=(const Function& v) = delete;

  /// Extract subfunction (view into the Function)
  ///
  /// @param i (std::size_t)
  ///         Index of subfunction.
  /// @returns    _Function_
  ///         The subfunction.
  Function sub(std::size_t i) const;

  /// Collapse a subfunction (view into the Function) to a stand-alone
  /// Function
  Function collapse() const;

  /// Return shared pointer to function space
  ///
  /// @returns _FunctionSpace_
  ///         Return the shared pointer.
  std::shared_ptr<const FunctionSpace> function_space() const;

  /// Return vector of expansion coefficients (non-const version)
  ///
  /// @returns  _PETScVector_
  ///         The vector of expansion coefficients.
  la::PETScVector& vector();

  /// Return vector of expansion coefficients (const version)
  ///
  /// @returns _PETScVector_
  ///         The vector of expansion coefficients (const).
  const la::PETScVector& vector() const;

  /// Interpolate function (on possibly non-matching meshes)
  ///
  /// @param    v (Function)
  ///         The function to be interpolated.
  void interpolate(const Function& v);

  /// Interpolate expression (on possibly non-matching meshes)
  ///
  /// @param    expr (Expression)
  ///         The expression to be interpolated.
  void interpolate(
      const std::function<void(
          Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>,
          const Eigen::Ref<const Eigen::Array<
              double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>)>& e);

  /// Interpolate expression (on possibly non-matching meshes)
  ///
  /// @param    expr (Expression)
  ///         The expression to be interpolated.
  // void interpolate(const Expression& e);

  /// Return value rank
  ///
  /// @returns int
  ///         The value rank.
  int value_rank() const;

  /// Return value size
  ///
  /// @returns std::size_t
  std::size_t value_size() const;

  /// Return value dimension for given axis
  ///
  /// @param    i (int)
  ///         The index of the axis.
  ///
  /// @returns    int
  ///         The value dimension.
  int value_dimension(int i) const;

  /// Return value shape
  ///
  /// @returns std::vector<std::size_t>
  ///         The value shape.
  std::vector<std::size_t> value_shape() const;

  /// Evaluate at given point in given cell
  ///
  /// @param    values (Eigen::Ref<Eigen::VectorXd>)
  ///         The values at the point.
  /// @param   x (Eigen::Ref<const Eigen::VectorXd>
  ///         The coordinates of the point.
  /// @param    cell (mesh::Cell)
  ///         The cell which contains the given point.
  void
  eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
           values,
       const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor>>
           x,
       const mesh::Cell& cell) const;

  /// Evaluate function at given coordinates
  ///
  /// @param    values (Eigen::Ref<Eigen::VectorXd> values)
  ///         The values.
  /// @param    x (Eigen::Ref<const Eigen::VectorXd> x)
  ///         The coordinates.
  void
  eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
           values,
       const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor>>
           x,
       const geometry::BoundingBoxTree& bb_tree) const;

  /// Restrict function to local cell (compute expansion coefficients w)
  ///
  /// @param    w (list of PetscScalars)
  ///         Expansion coefficients.
  /// @param    element (_FiniteElement_)
  ///         The element.
  /// @param    cell (_Cell_)
  ///         The cell.
  /// @param  coordinate_dofs (double *)
  ///         The coordinates
  void
  restrict(PetscScalar* w, const mesh::Cell& cell,
           const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs) const;

  /// Compute values at all mesh points
  ///
  /// @param    mesh (_mesh::Mesh_)
  ///         The mesh.
  /// @returns  point_values (EigenRowArrayXXd)
  ///         The values at all geometric points
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  compute_point_values(const mesh::Mesh& mesh) const;

  /// Compute values at all mesh points
  ///
  /// @returns    point_values (EigenRowArrayXXd)
  ///         The values at all geometric points
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  compute_point_values() const;

private:
  // The function space
  std::shared_ptr<const FunctionSpace> _function_space;

  // The vector of expansion coefficients (local)
  la::PETScVector _vector;
};
} // namespace function
} // namespace dolfin
