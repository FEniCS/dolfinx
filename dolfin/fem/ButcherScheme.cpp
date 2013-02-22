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
// Last changed: 2013-02-22

#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/function/FunctionAXPY.h>

#include "ButcherScheme.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ButcherScheme::ButcherScheme(std::vector<std::vector<boost::shared_ptr<const Form> > > stages, 
			     const FunctionAXPY& last_stage, 
			     std::vector<boost::shared_ptr<Function> > stage_solutions,
			     boost::shared_ptr<Function> u, 
			     boost::shared_ptr<Constant> t, 
			     boost::shared_ptr<Constant> dt) :
_stages(stages), _last_stage(last_stage), _stage_solutions(stage_solutions), 
  _u(u), _t(t), _dt(dt)
{}
//-----------------------------------------------------------------------------
ButcherScheme::ButcherScheme(std::vector<std::vector<boost::shared_ptr<const Form> > > stages, 
			     const FunctionAXPY& last_stage, 
			     std::vector<boost::shared_ptr<Function> > stage_solutions,
			     boost::shared_ptr<Function> u, 
			     boost::shared_ptr<Constant> t, 
			     boost::shared_ptr<Constant> dt, 
			     std::vector<boost::shared_ptr<const BoundaryCondition> > bcs) :
  _stages(stages), _last_stage(last_stage), _stage_solutions(stage_solutions), 
  _u(u), _t(t), _dt(dt), _bcs(bcs)
{}
//-----------------------------------------------------------------------------
std::vector<std::vector<boost::shared_ptr<const Form> > >& ButcherScheme::stages()
{
  return _stages;
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
std::vector<boost::shared_ptr<const BoundaryCondition> > ButcherScheme::bcs() const
{
  return _bcs;
}
//-----------------------------------------------------------------------------
