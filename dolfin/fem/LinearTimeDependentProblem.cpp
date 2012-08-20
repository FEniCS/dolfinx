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

#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/Function.h>
//#include "TensorProductForm.h"
#include "LinearTimeDependentProblem.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearTimeDependentProblem::
LinearTimeDependentProblem(const TensorProductForm& a,
                           const TensorProductForm& L,
                           Function& u)
  : Hierarchical<LinearTimeDependentProblem>(*this),
    _a(reference_to_no_delete_pointer(a)),
    _L(reference_to_no_delete_pointer(L)),
    _u(reference_to_no_delete_pointer(u))
{
  // Check forms
  check_forms();
}
//-----------------------------------------------------------------------------
LinearTimeDependentProblem::
LinearTimeDependentProblem(const TensorProductForm& a,
                           const TensorProductForm& L,
                           Function& u,
                           const BoundaryCondition& bc)
  : Hierarchical<LinearTimeDependentProblem>(*this),
    _a(reference_to_no_delete_pointer(a)),
    _L(reference_to_no_delete_pointer(L)),
    _u(reference_to_no_delete_pointer(u))
{
  // Store boundary condition
  _bcs.push_back(reference_to_no_delete_pointer(bc));

  // Check forms
  check_forms();
}
//-----------------------------------------------------------------------------
LinearTimeDependentProblem::
LinearTimeDependentProblem(const TensorProductForm& a,
                           const TensorProductForm& L,
                           Function& u,
                           std::vector<const BoundaryCondition*> bcs)
  : Hierarchical<LinearTimeDependentProblem>(*this),
    _a(reference_to_no_delete_pointer(a)),
    _L(reference_to_no_delete_pointer(L)),
    _u(reference_to_no_delete_pointer(u))
{
  // Store boundary conditions
  for (uint i = 0; i < bcs.size(); ++i)
    _bcs.push_back(reference_to_no_delete_pointer(*bcs[i]));

  // Check forms
  check_forms();
}
//-----------------------------------------------------------------------------
LinearTimeDependentProblem::
LinearTimeDependentProblem(boost::shared_ptr<const TensorProductForm> a,
                           boost::shared_ptr<const TensorProductForm> L,
                           boost::shared_ptr<Function> u,
                           std::vector<boost::shared_ptr<const BoundaryCondition> > bcs)
  : Hierarchical<LinearTimeDependentProblem>(*this),
    _a(a), _L(L), _u(u)
{
  // Store boundary conditions
  for (uint i = 0; i < bcs.size(); ++i)
    _bcs.push_back(bcs[i]);

  // Check forms
  check_forms();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const TensorProductForm>
LinearTimeDependentProblem::bilinear_form() const
{
  return _a;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const TensorProductForm>
LinearTimeDependentProblem::linear_form() const
{
  return _L;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Function> LinearTimeDependentProblem::solution()
{
  return _u;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const Function> LinearTimeDependentProblem::solution() const
{
  return _u;
}
//-----------------------------------------------------------------------------
std::vector<boost::shared_ptr<const BoundaryCondition> >
LinearTimeDependentProblem::bcs() const
{
  return _bcs;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const FunctionSpace>
LinearTimeDependentProblem::trial_space() const
{
  dolfin_assert(_u);
  return _u->function_space();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const FunctionSpace>
LinearTimeDependentProblem::test_space() const
{
  dolfin_assert(_a);
  return _a->function_space(0);
}
//-----------------------------------------------------------------------------
void LinearTimeDependentProblem::check_forms() const
{
  // Check rank of bilinear form a
  dolfin_assert(_a);
  if (_a->rank() != 2)
  {
    dolfin_error("LinearTimeDependentProblem.cpp",
                 "define linear variational problem a(u, v) == L(v) for all v",
                 "Expecting the left-hand side to be a bilinear form (not rank %d)",
                 _a->rank());
  }

  // Check rank of linear form L
  dolfin_assert(_L);
  if (_L->rank() != 1)
  {
    dolfin_error("LinearTimeDependentProblem.cpp",
                 "define linear variational problem a(u, v) = L(v) for all v",
                 "Expecting the right-hand side to be a linear form (not rank %d)",
                 _L->rank());
  }

  // Check that function space of solution variable matches trial space
  dolfin_assert(_a);
  dolfin_assert(_u);
  if (!_u->in(*_a->function_space(1)))
  {
    dolfin_error("LinearTimeDependentProblem.cpp",
                 "define linear variational problem a(u, v) = L(v) for all v",
                 "Expecting the solution variable u to be a member of the trial space");
  }
}
//-----------------------------------------------------------------------------
