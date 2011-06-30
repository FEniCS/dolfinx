// Copyright (C) 2008-2011 Anders Logg and Garth N. Wells
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
// Modified by Marie E. Rognes 2011
//
// First added:  2008-12-26
// Last changed: 2011-03-11

#ifndef __VARIATIONAL_PROBLEM_H
#define __VARIATIONAL_PROBLEM_H

#include <vector>
#include <boost/shared_ptr.hpp>

namespace dolfin
{

  class BoundaryCondition;
  class ErrorControl;
  class Form;
  class Function;
  class FunctionSpace;
  class GoalFunctional;
  template<class T> class MeshFunction;

  /// This class is deprecated and is only here to give an informative error
  /// message to users about the new interface.

  class VariationalProblem
  {
  public:

    /// Deprecated
    VariationalProblem(const Form& form_0,
                       const Form& form_1);

    /// Deprecated
    VariationalProblem(const Form& form_0,
                       const Form& form_1,
                       const BoundaryCondition& bc);

    /// Deprecated
    VariationalProblem(const Form& form_0,
                       const Form& form_1,
                       const std::vector<const BoundaryCondition*>& bcs);

    /// Deprecated
    VariationalProblem(boost::shared_ptr<const Form> form_0,
                       boost::shared_ptr<const Form> form_1,
                       std::vector<boost::shared_ptr<const BoundaryCondition> > bcs);

    /// Destructor
    ~VariationalProblem();

    /// Deprecated
    void solve(Function& u) const;

    /// Deprecated
    void solve(Function& u0, Function& u1) const;

    /// Deprecated
    void solve(Function& u0, Function& u1, Function& u2) const;

    /// Deprecated
    void solve(Function& u, const double tol, GoalFunctional& M) const;

    /// Deprecated
    void solve(Function& u, const double tol, Form& M, ErrorControl& ec) const;

  private:

    // Common error message
    void error_message() const;

  };

}

#endif
