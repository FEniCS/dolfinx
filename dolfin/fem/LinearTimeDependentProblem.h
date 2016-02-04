// Copyright (C) 2012 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2012-08-17
// Last changed: 2012-08-20

#ifndef __LINEAR_TIME_DEPENDENT_PROBLEM_H
#define __LINEAR_TIME_DEPENDENT_PROBLEM_H

#include <memory>
#include <dolfin/common/Hierarchical.h>

// FIXME: Temporary fix
#include "Form.h"

namespace dolfin
{

  // FIXME: Temporary fix
  typedef Form TensorProductForm;

  // Forward declarations
  class BoundaryCondition;

  /// This class represents a linear time-dependent variational problem:
  ///
  /// Find u in U = U_h (x) U_k such that
  ///
  ///     a(u, v) = L(v)  for all v in V = V_h (x) V_k,
  ///
  /// where U is a tensor-product trial space and V is a tensor-product
  /// test space.

  class LinearTimeDependentProblem : public Hierarchical<LinearTimeDependentProblem>
  {
  public:

    /// Create linear variational problem with a list of boundary
    /// conditions (shared pointer version)
    LinearTimeDependentProblem(std::shared_ptr<const TensorProductForm> a,
                               std::shared_ptr<const TensorProductForm> L,
                               std::shared_ptr<Function> u,
                               std::vector<std::shared_ptr<const BoundaryCondition>> bcs);

    /// Return bilinear form
    std::shared_ptr<const TensorProductForm> bilinear_form() const;

    /// Return linear form
    std::shared_ptr<const TensorProductForm> linear_form() const;

    /// Return solution variable
    std::shared_ptr<Function> solution();

    /// Return solution variable (const version)
    std::shared_ptr<const Function> solution() const;

    /// Return boundary conditions
    std::vector<std::shared_ptr<const BoundaryCondition>> bcs() const;

    /// Return trial space
    std::shared_ptr<const FunctionSpace> trial_space() const;

    /// Return test space
    std::shared_ptr<const FunctionSpace> test_space() const;

  private:

    // Check forms
    void check_forms() const;

    // The bilinear form
    std::shared_ptr<const TensorProductForm> _a;

    // The linear form
    std::shared_ptr<const TensorProductForm> _l;

    // The solution
    std::shared_ptr<Function> _u;

    // The boundary conditions
    std::vector<std::shared_ptr<const BoundaryCondition>> _bcs;

  };

}

#endif
