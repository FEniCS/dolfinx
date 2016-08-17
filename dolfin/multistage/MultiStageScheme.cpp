// Copyright (C) 2013 Johan Hake
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
// First added:  2013-02-15
// Last changed: 2014-10-13

#include <sstream>
#include <memory>
#include <dolfin/log/log.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/DirichletBC.h>

#include "MultiStageScheme.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiStageScheme::MultiStageScheme(
    std::vector<std::vector<std::shared_ptr<const Form>>> stage_forms,
    std::shared_ptr<const Form> last_stage,
    std::vector<std::shared_ptr<Function>> stage_solutions,
    std::shared_ptr<Function> u,
    std::shared_ptr<Constant> t,
    std::shared_ptr<Constant> dt,
    std::vector<double> dt_stage_offset,
    std::vector<int> jacobian_indices,
    unsigned int order,
    const std::string name,
    const std::string human_form,
    std::vector<std::shared_ptr<const DirichletBC>> bcs)
: Variable(name, ""), _stage_forms(stage_forms), _last_stage(last_stage),
  _stage_solutions(stage_solutions), _u(u), _t(t), _dt(dt),
  _dt_stage_offset(dt_stage_offset), _jacobian_indices(jacobian_indices),
  _order(order), _implicit(false), _human_form(human_form), _bcs(bcs)
{
  _check_arguments();
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::shared_ptr<const Form>>>&
MultiStageScheme::stage_forms()
{
  return _stage_forms;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Form> MultiStageScheme::last_stage()
{
  return _last_stage;
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<Function>>& MultiStageScheme::stage_solutions()
{
  return _stage_solutions;
}
//-----------------------------------------------------------------------------
std::shared_ptr<Function> MultiStageScheme::solution()
{
  return _u;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Function> MultiStageScheme::solution() const
{
  return _u;
}
//-----------------------------------------------------------------------------
std::shared_ptr<Constant> MultiStageScheme::t()
{
  return _t;
}
//-----------------------------------------------------------------------------
std::shared_ptr<Constant> MultiStageScheme::dt()
{
  return _dt;
}
//-----------------------------------------------------------------------------
const std::vector<double>& MultiStageScheme::dt_stage_offset() const
{
  return _dt_stage_offset;
}
//-----------------------------------------------------------------------------
unsigned int MultiStageScheme::order() const
{
  return _order;
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const DirichletBC>> MultiStageScheme::bcs() const
{
  return _bcs;
}
//-----------------------------------------------------------------------------
bool MultiStageScheme::implicit(unsigned int stage) const
{
  if (stage >= _stage_forms.size())
  {
    dolfin_error("MultiStageScheme.cpp",
     "querying if stage is implicit",
     "Expecting a stage less than the number of total stages in "
     "the scheme.");
  }

  return _stage_forms[stage].size() == 2;
}
//-----------------------------------------------------------------------------

bool MultiStageScheme::implicit() const
{
  return _implicit;
}
//-----------------------------------------------------------------------------
int MultiStageScheme::jacobian_index(unsigned int stage) const
{
  if (stage >= _jacobian_indices.size())
  {
    dolfin_error("MultiStageScheme.cpp",
     "querying for jacobian index",
     "Expecting a stage less than the number of total stages in "
     "the scheme.");
  }

  return _jacobian_indices[stage];
}
//-----------------------------------------------------------------------------
std::string MultiStageScheme::str(bool verbose) const
{
  if (!verbose)
    return name();

  std::stringstream s;
  s << name() << std::endl << _human_form;
  return s.str();
}
//-----------------------------------------------------------------------------
void MultiStageScheme::_check_arguments()
{
  // Check number of stage solutions is same as number of stage forms
  if (_stage_solutions.size()!=_stage_forms.size())
  {
    dolfin_error("MultiStageScheme.cpp",
                 "construct MultiStageScheme",
                 "Expecting the number of stage solutions to be the sames as "
                 "number of stage forms");
  }

  // Check that the number of coefficients in last form is the same as
  // number of stages
  /*
    if (_last_stage->num_coefficients() != _stage_forms.size())
    dolfin_error("MultiStageScheme.cpp",
       "construct MultiStageScheme",
       "Expecting the number of stage solutions to be the sames as " \
       "number of coefficients in the last form");
  */

  // Check solution is in the same space as the last stage solution
  if (!(_stage_solutions.size()==0 or
        _u->in(*_stage_solutions[_stage_solutions.size()-1]->function_space())))
  {
    dolfin_error("MultiStageScheme.cpp",
                 "construct MultiStageScheme",
                 "Expecting all solutions to be in the same FunctionSpace");
  }

  // Check number of passed stage forms
  for (unsigned int i=0; i < _stage_forms.size();i++)
  {
    // Check solution is in the same space as the stage solution
    if (!_u->in(*_stage_solutions[i]->function_space()))
    {
      dolfin_error("MultiStageScheme.cpp",
                   "construct MultiStageScheme",
                   "Expecting all solutions to be in the same FunctionSpace");
    }

    // Check we have correct number of forms
    if (_stage_forms[i].size()==0 or _stage_forms[i].size()>2)
    {
      dolfin_error("MultiStageScheme.cpp",
                   "construct MultiStageScheme",
                   "Expecting stage_forms to only include vectors of size 1 or 2");
    }

    // Check if Scheme is implicit
    if (_stage_forms[i].size()==2)
    {
      _implicit = true;

      // First form should be the linear (in testfunction) form
      if (_stage_forms[i][0]->rank() != 1)
      {
        dolfin_error("MultiStageScheme.cpp",
                     "construct MultiStageScheme",
                     "Expecting the right-hand side of stage form %d to be a "
                     "linear form (not rank %d)", i, _stage_forms[i][0]->rank());
      }

      // Second form should be the bilinear form
      if (_stage_forms[i][1]->rank() != 2)
      {
        dolfin_error("MultiStageScheme.cpp",
                     "construct MultiStageScheme",
                     "Expecting the left-hand side of stage form %d to be a "
                     "linear form (not rank %d)", i, _stage_forms[i][1]->rank());
      }

      // Check that function space of solution variable matches trial space
      if (!_stage_solutions[i]->in(*_stage_forms[i][1]->function_space(1)))
      {
        dolfin_error("MultiStageScheme.cpp",
                     "construct MultiStageScheme",
                     "Expecting the stage solution %d to be a member of the "
                     "trial space of stage form %d", i, i);
      }

      // Check that function spaces of bcs are contained in trial space
      for (const auto bc: _bcs)
      {
        if (!_stage_forms[i][1]->function_space(1)->contains(*bc->function_space()))
        {
          dolfin_error("MultiStageScheme.cpp",
                       "construct MultiStageScheme",
                       "Expecting the boundary conditions to to live on (a "
                       "subspace of) the trial space");
        }
      }
    }
    else
    {
      // Check explicit stage form
      if (_stage_forms[i][0]->rank() != 1)
      {
        dolfin_error("MultiStageScheme.cpp",
                     "construct MultiStageScheme",
                     "Expecting stage form %d to be a linear form (not "
                     "rank %d)", i, _stage_forms[i][0]->rank());
      }

      // Check that function space of solution variable matches trial space
      if (!_stage_solutions[i]->in(*_stage_forms[i][0]->function_space(0)))
      {
        dolfin_error("MultiStageScheme.cpp",
                     "construct MultiStageScheme",
                     "Expecting the stage solution %d to be a member of the "
                     "test space of stage form %d", i, i);
      }

      // Check that function spaces of bcs are contained in trial space
      for (const auto bc: _bcs)
      {
        if (!_stage_forms[i][0]->function_space(0)->contains(*bc->function_space()))
        {
          dolfin_error("MultiStageScheme.cpp",
                       "construct MultiStageScheme",
                       "Expecting the boundary conditions to to live on (a "
                       "subspace of) the trial space");
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
