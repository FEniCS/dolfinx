// Copyright (C) 2003-2005 Anders Logg
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
// First added:  2003-11-20
// Last changed: 2005

#include <dolfin/log/dolfin_log.h>
#include "TimeSlab.h"
#include "Sample.h"
#include "ODE.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Sample::Sample(TimeSlab& timeslab, real t,
	       std::string name, std::string label) :
  Variable(name, label), timeslab(timeslab), time(t)
{
  // Prepare time slab for sample
  timeslab.sample(t);
}
//-----------------------------------------------------------------------------
Sample::~Sample()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
unsigned int Sample::size() const
{
  return timeslab.size();
}
//-----------------------------------------------------------------------------
real Sample::t() const
{
  return timeslab.ode.time(time);
}
//-----------------------------------------------------------------------------
real Sample::u(unsigned int index) const
{
  return timeslab.usample(index, time);
}
//-----------------------------------------------------------------------------
real Sample::k(unsigned int index) const
{
  return timeslab.ksample(index, time);
}
//-----------------------------------------------------------------------------
real Sample::r(unsigned int index) const
{
  return timeslab.rsample(index, time);
}
//-----------------------------------------------------------------------------
