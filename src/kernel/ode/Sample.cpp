// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Solution.h>
#include <dolfin/Sample.h>

using namespace dolfin;


/*
//-----------------------------------------------------------------------------
Sample::Sample(Solution& solution, RHS& f, real t) :
  solution(solution), f(f), time(t)
{
  // Set name and label (same as solution)
  rename(solution.name(), solution.label());
}
//-----------------------------------------------------------------------------
Sample::~Sample()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
unsigned int Sample::size() const
{
  return solution.size();
}
//-----------------------------------------------------------------------------
real Sample::t() const
{
  return time;
}
//-----------------------------------------------------------------------------
real Sample::u(unsigned int index)
{
  return solution.u(index, 0, time);
}
//-----------------------------------------------------------------------------
real Sample::k(unsigned int index)
{
  return solution.k(index, time);
}
//-----------------------------------------------------------------------------
real Sample::r(unsigned int index)
{
  return solution.r(index, time, f);
}
//-----------------------------------------------------------------------------
*/
