// Copyright (C) 2011 Anders Logg
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
// First added:  2011-06-22
// Last changed: 2011-06-22

#ifndef __LINEAR_VARIATIONAL_PROBLEM_H
#define __LINEAR_VARIATIONAL_PROBLEM_H

#include <boost/shared_ptr.hpp>
#include <dolfin/common/Hierarchical.h>

namespace dolfin
{

  // Forward declarations
  class Form;
  class Function;
  class FunctionSpace;
  class BoundaryCondition;

  /// This class represents a linear variational problem:
  ///
  /// Find u in V such that
  ///
  ///     a(u, v) = L(v)  for all v in V^,
  ///
  /// where V is the trial space and V^ is the test space.

  class LinearVariationalProblem : public Hierarchical<LinearVariationalProblem>
  {
  public:

    /// Create linear variational problem with a single boundary condition
    LinearVariationalProblem(const Form& a,
                             const Form& L,
                             Function& u,
                             const BoundaryCondition& bc);

    /// Create linear variational problem with a single boundary condition
    /// (shared pointer version)
    LinearVariationalProblem(boost::shared_ptr<const Form> a,
                             boost::shared_ptr<const Form> L,
                             boost::shared_ptr<Function> u,
                             boost::shared_ptr<const BoundaryCondition> bc);

    /// Create linear variational problem with a list of boundary conditions
    LinearVariationalProblem(const Form& a,
                             const Form& L,
                             Function& u,
                             std::vector<const BoundaryCondition*> bcs);

    /// Create linear variational problem with a list of boundary conditions
    /// (shared pointer version)
    LinearVariationalProblem(boost::shared_ptr<const Form> a,
                             boost::shared_ptr<const Form> L,
                             boost::shared_ptr<Function> u,
                             std::vector<boost::shared_ptr<const BoundaryCondition> > bcs);

    /// Return bilinear form
    boost::shared_ptr<const Form> bilinear_form() const;

    /// Return linear form
    boost::shared_ptr<const Form> linear_form() const;

    /// Return solution variable
    boost::shared_ptr<Function> solution();

    /// Return solution variable (const version)
    boost::shared_ptr<const Function> solution() const;

    /// Return boundary conditions
    std::vector<boost::shared_ptr<const BoundaryCondition> > bcs() const;

    /// Return trial space
    boost::shared_ptr<const FunctionSpace> trial_space() const;

    /// Return test space
    boost::shared_ptr<const FunctionSpace> test_space() const;

  private:

    // Check forms
    void check_forms() const;

    // The bilinear form
    boost::shared_ptr<const Form> _a;

    // The linear form
    boost::shared_ptr<const Form> _L;

    // The solution
    boost::shared_ptr<Function> _u;

    // The boundary conditions
    std::vector<boost::shared_ptr<const BoundaryCondition> > _bcs;

  };

}

#endif
