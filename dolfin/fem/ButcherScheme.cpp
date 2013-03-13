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
// Last changed: 2013-03-13

#include <sstream>
#include <boost/shared_ptr.hpp>
#include <dolfin/function/FunctionAXPY.h>
#include <dolfin/log/log.h>

#include "ButcherScheme.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ButcherScheme::ButcherScheme(std::vector<std::vector<boost::shared_ptr<const Form> > > stage_forms, 
			     const FunctionAXPY& last_stage, 
			     std::vector<boost::shared_ptr<Function> > stage_solutions,
			     boost::shared_ptr<Function> u, 
			     boost::shared_ptr<Constant> t, 
			     boost::shared_ptr<Constant> dt,
			     std::vector<double> dt_stage_offset, 
			     unsigned int order,
			     const std::string name,
			     const std::string human_form) : 
  Variable(name, ""), _stage_forms(stage_forms), _last_stage(last_stage), 
  _stage_solutions(stage_solutions), _u(u), _t(t), _dt(dt), 
  _dt_stage_offset(dt_stage_offset), _order(order), _implicit(false), 
  _human_form(human_form)
{
  _check_arguments();
}
//-----------------------------------------------------------------------------
ButcherScheme::ButcherScheme(std::vector<std::vector<boost::shared_ptr<const Form> > > stage_forms, 
			     const FunctionAXPY& last_stage, 
			     std::vector<boost::shared_ptr<Function> > stage_solutions,
			     boost::shared_ptr<Function> u, 
			     boost::shared_ptr<Constant> t, 
			     boost::shared_ptr<Constant> dt,
			     std::vector<double> dt_stage_offset, 
			     unsigned int order,
			     const std::string name,
			     const std::string human_form,
			     std::vector<const DirichletBC* > bcs) :
  Variable(name, ""), _stage_forms(stage_forms), _last_stage(last_stage), 
  _stage_solutions(stage_solutions), _u(u), _t(t), _dt(dt), 
  _dt_stage_offset(dt_stage_offset), _order(order), _implicit(false), 
  _human_form(human_form), 
  _bcs(bcs)
{
  _check_arguments();
}
//-----------------------------------------------------------------------------
std::vector<std::vector<boost::shared_ptr<const Form> > >& ButcherScheme::stage_forms()
{
  return _stage_forms;
}
//-----------------------------------------------------------------------------
FunctionAXPY& ButcherScheme::last_stage()
{
  return _last_stage;
}
//-----------------------------------------------------------------------------
std::vector<boost::shared_ptr<Function> >& ButcherScheme::stage_solutions()
{
  return _stage_solutions;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Function> ButcherScheme::solution()
{
  return _u;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const Function> ButcherScheme::solution() const
{
  return _u;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Constant> ButcherScheme::t()
{
  return _t;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Constant> ButcherScheme::dt()
{
  return _dt;
}
//-----------------------------------------------------------------------------
const std::vector<double>& ButcherScheme::dt_stage_offset() const
{
  return _dt_stage_offset;
}
//-----------------------------------------------------------------------------
unsigned int ButcherScheme::order() const
{
  return _order;
}
//-----------------------------------------------------------------------------
std::vector<const DirichletBC* > ButcherScheme::bcs() const
{
  return _bcs;
}
//-----------------------------------------------------------------------------
bool ButcherScheme::implicit(unsigned int stage) const
{
  dolfin_assert(stage < _stage_forms.size());
  return _stage_forms[stage].size() == 2;
}
//-----------------------------------------------------------------------------
bool ButcherScheme::implicit() const
{
  return _implicit;
}
//-----------------------------------------------------------------------------
std::string ButcherScheme::str(bool verbose) const
{
  if (!verbose)
    return name();

  std::stringstream s;
  s << name() << std::endl << _human_form;
  return s.str();
}
//-----------------------------------------------------------------------------
void ButcherScheme::_check_arguments()
{
  // Check number of stage sollutions is same as number of stage forms
  if (_stage_solutions.size()!=_stage_forms.size())
  {
    dolfin_error("ButcherScheme.cpp",
		 "constructing ButcherScheme",
		 "Expecting the number of stage solutions to be the sames as "\
		 "number of stage forms");
  }

  // Check solution is in the same space as the last stage solution
  if (!_u->in(*_stage_solutions[_stage_solutions.size()-1]->function_space()))
  {
    dolfin_error("ButcherScheme.cpp",
		 "constructing ButcherScheme",
		 "Expecting all solutions to be in the same FunctionSpace");
  }

  // Check number of passed stage forms
  for (unsigned int i=0; i < _stage_forms.size();i++)
  {
    
    // Check solution is in the same space as the stage solution
    if (!_u->in(*_stage_solutions[i]->function_space()))
    {
      dolfin_error("ButcherScheme.cpp",
		   "constructing ButcherScheme",
		   "Expecting all solutions to be in the same FunctionSpace");
    }

    // Check we have correct number of forms
    if (_stage_forms[i].size()==0 || _stage_forms[i].size()>2)
    {
      dolfin_error("ButcherScheme.cpp",
		   "constructing ButcherScheme",
		   "Expecting stage_forms to only include vectors of size 1 or 2");
    }
    
    // Check if Scheme is implicit
    if (_stage_forms[i].size()==2)
    {
      _implicit = true;

      // First form should be the linear (in testfunction) form
      if (_stage_forms[i][0]->rank() != 1)
      {
	dolfin_error("ButcherScheme.cpp",
		     "constructing ButcherScheme",
		     "Expecting the right-hand side of stage form %d to be a "\
		     "linear form (not rank %d)", i, _stage_forms[i][0]->rank());
      }

      // Second form should be the bilinear form
      if (_stage_forms[i][1]->rank() != 2)
      {
	dolfin_error("ButcherScheme.cpp",
		     "constructing ButcherScheme",
		     "Expecting the left-hand side of stage form %d to be a "\
		     "linear form (not rank %d)", i, _stage_forms[i][1]->rank());
      }
      
      // Check that function space of solution variable matches trial space
      if (!_stage_solutions[i]->in(*_stage_forms[i][1]->function_space(1)))
      {
	dolfin_error("ButcherScheme.cpp",
		     "constructing ButcherScheme",
		     "Expecting the stage solution %d to be a member of the "
		     "trial space of stage form %d", i, i);
      }
      
    }
    else
    {

      // Check explicit stage form 
      if (_stage_forms[i][0]->rank() != 1)
      {
	dolfin_error("ButcherScheme.cpp",
		     "constructing ButcherScheme",
		     "Expecting stage form %d to be a linear form (not "\
		     "rank %d)", i, _stage_forms[i][0]->rank());
      }

      // Check that function space of solution variable matches trial space
      if (!_stage_solutions[i]->in(*_stage_forms[i][0]->function_space(0)))
      {
	dolfin_error("ButcherScheme.cpp",
		     "constructing ButcherScheme",
		     "Expecting the stage solution %d to be a member of the "
		     "test space of stage form %d", i, i);
      }
      
    }

  }
}
//-----------------------------------------------------------------------------
