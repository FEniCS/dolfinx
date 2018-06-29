// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "GenericFunction.h"
#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <memory>
#include <petscsys.h>
#include <vector>

namespace dolfin
{
namespace la
{
class PETScVector;
}

namespace mesh
{
class Cell;
class Mesh;
} // namespace mesh

namespace function
{
class FunctionSpace;

/// This class represents a function \f$ u_h \f$ in a finite
/// element function space \f$ V_h \f$, given by
///
/// \f[     u_h = \sum_{i=1}^{n} U_i \phi_i \f]
/// where \f$ \{\phi_i\}_{i=1}^{n} \f$ is a basis for \f$ V_h \f$,
/// and \f$ U \f$ is a vector of expansion coefficients for \f$ u_h \f$.

class Function : public GenericFunction
{
public:
  Function() {}

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
  /// @param x (_GenericVector_)
  ///         The vector.
  Function(std::shared_ptr<const FunctionSpace> V,
           std::shared_ptr<la::PETScVector> x);

  /// Copy constructor
  ///
  /// If v is not a sub-function, the new Function shares the
  /// FunctionSpace of v and copies the degree-of-freedom vector. If v
  /// is a sub-Function, the new Function is a collapsed version of v.
  ///
  /// @param v (_Function_)
  ///         The object to be copied.
  Function(const Function& v);

  /// Destructor
  virtual ~Function() = default;

  // Assignment from function
  //
  // @param v (_Function_)
  //         Another function.
  // const Function& operator= (const Function& v);

  /// Extract subfunction (view into the Function)
  ///
  /// @param i (std::size_t)
  ///         Index of subfunction.
  /// @returns    _Function_
  ///         The subfunction.
  Function sub(std::size_t i) const;

  /// Return shared pointer to function space
  ///
  /// @returns _FunctionSpace_
  ///         Return the shared pointer.
  virtual std::shared_ptr<const FunctionSpace> function_space() const override
  {
    assert(_function_space);
    return _function_space;
  }

  /// Return vector of expansion coefficients (non-const version)
  ///
  /// @returns  _PETScVector_
  ///         The vector of expansion coefficients.
  std::shared_ptr<la::PETScVector> vector();

  /// Return vector of expansion coefficients (const version)
  ///
  /// @returns _PETScVector_
  ///         The vector of expansion coefficients (const).
  std::shared_ptr<const la::PETScVector> vector() const;

  /// Evaluate function at given coordinates
  ///
  /// @param    values (Eigen::Ref<Eigen::VectorXd> values)
  ///         The values.
  /// @param    x (Eigen::Ref<const Eigen::VectorXd> x)
  ///         The coordinates.
  void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                values,
            Eigen::Ref<const EigenRowArrayXXd> x) const override;

  /// Interpolate function (on possibly non-matching meshes)
  ///
  /// @param    v (GenericFunction)
  ///         The function to be interpolated.
  void interpolate(const GenericFunction& v);

  //--- Implementation of GenericFunction interface ---

  /// Return value rank
  ///
  /// @returns std::size_t
  ///         The value rank.
  virtual std::size_t value_rank() const override;

  /// Return value dimension for given axis
  ///
  /// @param    i (std::size_t)
  ///         The index of the axis.
  ///
  /// @returns    std::size_t
  ///         The value dimension.
  virtual std::size_t value_dimension(std::size_t i) const override;

  /// Return value shape
  ///
  /// @returns std::vector<std::size_t>
  ///         The value shape.
  virtual std::vector<std::size_t> value_shape() const override;

  /// Evaluate at given point in given cell
  ///
  /// @param    values (Eigen::Ref<Eigen::VectorXd>)
  ///         The values at the point.
  /// @param   x (Eigen::Ref<const Eigen::VectorXd>
  ///         The coordinates of the point.
  /// @param    cell (mesh::Cell)
  ///         The cell which contains the given point.
  virtual void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor>>
                        values,
                    Eigen::Ref<const EigenRowArrayXXd> x,
                    const mesh::Cell& cell) const override;

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
  virtual void restrict(
      PetscScalar* w, const fem::FiniteElement& element, const mesh::Cell& cell,
      const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs) const override;

  /// Compute values at all mesh points
  ///
  /// @param    mesh (_mesh::Mesh_)
  ///         The mesh.
  /// @returns  point_values (EigenRowArrayXXd)
  ///         The values at all geometric points
  virtual Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>
  compute_point_values(const mesh::Mesh& mesh) const override;

  /// Compute values at all mesh points
  ///
  /// @returns    point_values (EigenRowArrayXXd)
  ///         The values at all geometric points
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  compute_point_values() const;

private:
  // Initialize vector
  void init_vector();

  // The function space
  std::shared_ptr<const FunctionSpace> _function_space;

  // The vector of expansion coefficients (local)
  std::shared_ptr<la::PETScVector> _vector;
};
} // namespace function
} // namespace dolfin
