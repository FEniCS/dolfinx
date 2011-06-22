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
// Last changed: 2011-06-21

#include <dolfin/log/log.h>
#include "VariationalProblem.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& form_0,
                                       const Form& form_1)
{
  error_message();
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& form_0,
                                       const Form& form_1,
                                       const BoundaryCondition& bc)
{
  error_message();
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& form_0,
                                       const Form& form_1,
                                       const std::vector<const BoundaryCondition*>& bcs)
{
  error_message();
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(boost::shared_ptr<const Form> form_0,
                                       boost::shared_ptr<const Form> form_1,
                                       std::vector<boost::shared_ptr<const BoundaryCondition> > bcs)
{
  error_message();
}
//-----------------------------------------------------------------------------
VariationalProblem::~VariationalProblem() {}
void VariationalProblem::solve(Function& u) const {}
void VariationalProblem::solve(Function& u0, Function& u1) const {}
void VariationalProblem::solve(Function& u0, Function& u1, Function& u2) const {}
void VariationalProblem::solve(Function& u, const double tolerance, GoalFunctional& M) const {}
void VariationalProblem::solve(Function& u, const double tolerance, Form& M, ErrorControl& ec) const {}
//-----------------------------------------------------------------------------
void VariationalProblem::error_message() const
{
  dolfin_error("VariationalProblem.cpp",
               "create variational problem",
               "the VariationalProblem class has been removed, use solve(a == L)");
}
//-----------------------------------------------------------------------------
