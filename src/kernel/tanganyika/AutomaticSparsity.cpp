// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/Vector.h>
#include <dolfin/AutomaticSparsity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
AutomaticSparsity::AutomaticSparsity(int N, ODE& ode) : GenericSparsity(N)
{
  list = new ShortList<int>[N];
  increment = dolfin_get("sparsity check increment");

  computeSparsity(ode);
}
//-----------------------------------------------------------------------------
AutomaticSparsity::~AutomaticSparsity()
{
  if ( list )
    delete list;
  list = 0;
}
//-----------------------------------------------------------------------------
GenericSparsity::Type AutomaticSparsity::type() const
{
  return automatic;
}
//-----------------------------------------------------------------------------
AutomaticSparsity::Iterator* AutomaticSparsity::createIterator(int i) const
{
  dolfin_assert(i >= 0);
  dolfin_assert(i < N);

  return new Iterator(i, *this);
}
//-----------------------------------------------------------------------------
void AutomaticSparsity::computeSparsity(ODE& ode)
{
  // Randomize solution vector
  Vector u(N);
  u.rand();

  // Check dependencies for all components
  Progress p("Computing sparsity", N);
  int sum = 0;
  for (int i = 0; i < N; i++) {
    
    // Count the number of dependencies
    int size = 0;
    real f0 = ode.f(u, 0.0, i);
    for (int j = 0; j < N; j++)
      if ( checkdep(ode, u, f0, i, j) )
	size++;
    
    // Compute total number of dependencies
    sum += size;

    // Allocate list
    list[i].init(size);

    // Set the dependencies
    int pos = 0;
    for (int j = 0; j < N; j++)
      if ( checkdep(ode, u, f0, i, j) )
	list[i](pos++) = j;

    // Update progress
    p = i;
  }

  dolfin_info("Automatically detected %d dependencies.", sum);
}
//-----------------------------------------------------------------------------
bool AutomaticSparsity::checkdep(ODE& ode, Vector& u, real f0, int i, int j)
{
  // Save original value
  real uj = u(j);

  // Change value and compute new value for f_i
  u(j) += increment;
  real f = ode.f(u, 0.0, i);

  // Restore the value
  u(j) = uj;

  // Compare function values
  return fabs(f - f0) > DOLFIN_EPS;
}
//-----------------------------------------------------------------------------
AutomaticSparsity::Iterator::Iterator(int i, const AutomaticSparsity& sparsity)
  : GenericSparsity::Iterator(i), s(sparsity)
{
  pos = 0;

  if ( s.list[i].size() == 0 )
    at_end = true;
  else if ( s.list[i](0) == -1 )
    at_end = true;
  else
    at_end = false;
}
//-----------------------------------------------------------------------------
AutomaticSparsity::Iterator::~Iterator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
AutomaticSparsity::Iterator& AutomaticSparsity::Iterator::operator++()
{
  if ( pos == (s.list[i].size() - 1) )
    at_end = true;
  else if ( s.list[i](pos+1) == -1 )
    at_end = true;
  else
    pos++;
}
//-----------------------------------------------------------------------------
int AutomaticSparsity::Iterator::operator*() const
{
  return s.list[i](pos);
}
//-----------------------------------------------------------------------------
bool AutomaticSparsity::Iterator::end() const
{
  return at_end;
}
//-----------------------------------------------------------------------------
