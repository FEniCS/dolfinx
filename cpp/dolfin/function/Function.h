// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FunctionAXPY.h"
#include "GenericFunction.h"
#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ufc
{
// Forward declarations
class cell;
}

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
}

namespace function
{
class Expression;
class FunctionSpace;

/// This class represents a function :math:`u_h` in a finite
/// element function space :math:`V_h`, given by
///
/// .. math::
///
///     u_h = \sum_{i=1}^{n} U_i \phi_i
///
/// where :math:`\{\phi_i\}_{i=1}^{n}` is a basis for :math:`V_h`,
/// and :math:`U` is a vector of expansion coefficients for :math:`u_h`.

class Function : public GenericFunction
{
public:
  Function() {}

  /// Create function on given function space
  ///
  /// *Arguments*
  ///     V (_FunctionSpace_)
  ///         The function space.
  explicit Function(std::shared_ptr<const FunctionSpace> V);

  /// Create function on given function space with a given vector
  ///
  /// *Warning: This constructor is intended for internal library use only*
  ///
  /// *Arguments*
  ///     V (_FunctionSpace_)
  ///         The function space.
  ///     x (_GenericVector_)
  ///         The vector.
  Function(std::shared_ptr<const FunctionSpace> V,
           std::shared_ptr<la::PETScVector> x);

  /// Copy constructor
  ///
  /// If v is not a sub-function, the new Function shares the
  /// FunctionSpace of v and copies the degree-of-freedom vector. If v
  /// is a sub-Function, the new Function is a collapsed version of v.
  ///
  /// *Arguments*
  ///     v (_Function_)
  ///         The object to be copied.
  Function(const Function& v);

  /// Destructor
  virtual ~Function();

  /// Assignment from function
  ///
  /// *Arguments*
  ///     v (_Function_)
  ///         Another function.
  // const Function& operator= (const Function& v);

  /// Assignment from expression using interpolation
  ///
  /// *Arguments*
  ///     v (_Expression_)
  ///         The expression.
  const Function& operator=(const Expression& v);

  /// Assignment from linear combination of function
  ///
  /// *Arguments*
  ///     v (_FunctionAXPY_)
  ///         A linear combination of other Functions
  void operator=(const function::FunctionAXPY& axpy);

  /// Extract subfunction (view into the Function)
  ///
  /// *Arguments*
  ///     i (std::size_t)
  ///         Index of subfunction.
  /// *Returns*
  ///     _Function_
  ///         The subfunction.
  Function sub(std::size_t i) const;

  /// Return shared pointer to function space
  ///
  /// *Returns*
  ///     _FunctionSpace_
  ///         Return the shared pointer.
  virtual std::shared_ptr<const FunctionSpace> function_space() const override
  {
    dolfin_assert(_function_space);
    return _function_space;
  }

  /// Return vector of expansion coefficients (non-const version)
  ///
  /// *Returns*
  ///     _GenericVector_
  ///         The vector of expansion coefficients.
  std::shared_ptr<la::PETScVector> vector();

  /// Return vector of expansion coefficients (const version)
  ///
  /// *Returns*
  ///     _GenericVector_
  ///         The vector of expansion coefficients (const).
  std::shared_ptr<const la::PETScVector> vector() const;

  /// Evaluate function at given coordinates
  ///
  /// @param    values (Eigen::Ref<Eigen::VectorXd> values)
  ///         The values.
  /// @param    x (Eigen::Ref<const Eigen::VectorXd> x)
  ///         The coordinates.
  void eval(Eigen::Ref<EigenRowMatrixXd> values,
            Eigen::Ref<const EigenRowMatrixXd> x) const override;

  /// Evaluate function at given coordinates in given cell
  ///
  /// *Arguments*
  /// @param    values (Eigen::Ref<Eigen::VectorXd>)
  ///         The values.
  /// @param    x (Eigen::Ref<const Eigen::VectorXd>)
  ///         The coordinates.
  /// @param    dolfin_cell (_Cell_)
  ///         The cell.
  /// @param    ufc_cell (ufc::cell)
  ///         The ufc::cell.
  void eval(Eigen::Ref<EigenRowMatrixXd> values,
            Eigen::Ref<const EigenRowMatrixXd> x, const mesh::Cell& dolfin_cell,
            const ufc::cell& ufc_cell) const;

  /// Interpolate function (on possibly non-matching meshes)
  ///
  /// @param    v (GenericFunction)
  ///         The function to be interpolated.
  void interpolate(const GenericFunction& v);

  /// Extrapolate function (from a possibly lower-degree function space)
  ///
  /// *Arguments*
  ///     v (_Function_)
  ///         The function to be extrapolated.
  void extrapolate(const Function& v);

  //--- Implementation of GenericFunction interface ---

  /// Return value rank
  ///
  /// *Returns*
  ///     std::size_t
  ///         The value rank.
  virtual std::size_t value_rank() const override;

  /// Return value dimension for given axis
  ///
  /// *Arguments*
  ///     i (std::size_t)
  ///         The index of the axis.
  ///
  /// *Returns*
  ///     std::size_t
  ///         The value dimension.
  virtual std::size_t value_dimension(std::size_t i) const override;

  /// Return value shape
  ///
  /// *Returns*
  ///     std::vector<std::size_t>
  ///         The value shape.
  virtual std::vector<std::size_t> value_shape() const override;

  /// Evaluate at given point in given cell
  ///
  /// @param    values (Eigen::Ref<Eigen::VectorXd>)
  ///         The values at the point.
  /// @param   x (Eigen::Ref<const Eigen::VectorXd>
  ///         The coordinates of the point.
  /// @param    cell (ufc::cell)
  ///         The cell which contains the given point.
  virtual void eval(Eigen::Ref<EigenRowMatrixXd> values,
                    Eigen::Ref<const EigenRowMatrixXd> x,
                    const ufc::cell& cell) const override;

  /// Restrict function to local cell (compute expansion coefficients w)
  ///
  /// @param    w (list of doubles)
  ///         Expansion coefficients.
  /// @param    element (_FiniteElement_)
  ///         The element.
  /// @param    dolfin_cell (_Cell_)
  ///         The cell.
  /// @param  coordinate_dofs (double *)
  ///         The coordinates
  /// @param    ufc_cell (ufc::cell).
  ///         The ufc::cell.
  virtual void restrict(double* w, const fem::FiniteElement& element,
                        const mesh::Cell& dolfin_cell,
                        const double* coordinate_dofs,
                        const ufc::cell& ufc_cell) const override;

  /// Compute values at all mesh vertices
  ///
  /// @param    vertex_values (Array<double>)
  ///         The values at all vertices.
  /// @param    mesh (_mesh::Mesh_)
  ///         The mesh.
  virtual EigenRowArrayXXd
  compute_vertex_values(const mesh::Mesh& mesh) const override;

  /// Compute values at all mesh vertices
  ///
  /// @param    vertex_values (Array<double>)
  ///         The values at all vertices.
  EigenRowArrayXXd compute_vertex_values() const;

  /// Allow extrapolation when evaluating the Function
  ///
  /// @param allow_extrapolation (bool)
  ///         Whether or not permit extrapolation.
  void set_allow_extrapolation(bool allow_extrapolation)
  {
    _allow_extrapolation = allow_extrapolation;
  }

  /// Check if extrapolation is permitted when evaluating the Function
  ///
  /// @return bool
  ///         True if extrapolation is permitted, otherwise false
  bool get_allow_extrapolation() const { return _allow_extrapolation; }

private:
  // Friends
  friend class FunctionAssigner;

  // Initialize vector
  void init_vector();

  // The function space
  std::shared_ptr<const FunctionSpace> _function_space;

  // The vector of expansion coefficients (local)
  std::shared_ptr<la::PETScVector> _vector;

  // True if extrapolation should be allowed
  bool _allow_extrapolation;
};
}
}
