// Copyright (C) 2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/Variable.h>
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
class Mesh;
class Cell;
} // namespace mesh

namespace function
{
class FunctionSpace;

/// This is a common base class for functions. Functions can be
/// evaluated at a given point and they can be restricted to a given
/// cell in a finite element mesh. This functionality is implemented
/// by sub-classes that implement the eval() and restrict()
/// functions.
///
/// DOLFIN provides two implementations of the GenericFunction
/// interface in the form of the classes Function and Expression.
///
/// Sub-classes may optionally implement the update() function that
/// will be called prior to restriction when running in parallel.

class GenericFunction : public common::Variable
{
public:
  /// Constructor
  GenericFunction();

  /// Destructor
  virtual ~GenericFunction();

  //--- Functions that must be implemented by sub-classes ---

  /// Return value rank
  virtual std::size_t value_rank() const = 0;

  /// Return value dimension for given axis
  virtual std::size_t value_dimension(std::size_t i) const = 0;

  /// Return value shape
  virtual std::vector<std::size_t> value_shape() const = 0;

  /// Evaluate at given point in given cell
  virtual void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor>>
                        values,
                    Eigen::Ref<const EigenRowArrayXXd> x,
                    const mesh::Cell& cell) const;

  /// Evaluate at given point
  virtual void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor>>
                        values,
                    Eigen::Ref<const EigenRowArrayXXd> x) const;

  /// Restrict function to local cell (compute expansion coefficients w)
  virtual void
  restrict(PetscScalar* w, const fem::FiniteElement& element,
           const mesh::Cell& cell,
           const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs) const = 0;

  /// Compute values at all mesh vertices
  virtual Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>
  compute_point_values(const mesh::Mesh& mesh) const = 0;

  //--- Optional functions to be implemented by sub-classes ---

  /// Update off-process ghost coefficients
  virtual void update() const {}

  //--- Convenience functions ---

  /// Evaluation at given point

  /// Return value size (product of value dimensions)
  std::size_t value_size() const;

  /// Pointer to FunctionSpace, if appropriate, otherwise NULL
  virtual std::shared_ptr<const FunctionSpace> function_space() const = 0;
};
} // namespace function
} // namespace dolfin
