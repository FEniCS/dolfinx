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
AutomaticSparsity::AutomaticSparsity(unsigned int N, ODE& ode) : GenericSparsity(N)
{
  list = new Array<unsigned int>[N];
  increment = dolfin_get("sparsity check increment");

  computeSparsity(ode);
}
//-----------------------------------------------------------------------------
AutomaticSparsity::~AutomaticSparsity()
{
  if ( list )
    delete [] list;
  list = 0;
}
//-----------------------------------------------------------------------------
GenericSparsity::Type AutomaticSparsity::type() const
{
  return automatic;
}
//-----------------------------------------------------------------------------
AutomaticSparsity::Iterator* AutomaticSparsity::createIterator(unsigned int i) const
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
  unsigned int sum = 0;
  for (unsigned int i = 0; i < N; i++) {
    
    // Count the number of dependencies
    unsigned int size = 0;
    real f0 = ode.f(u, 0.0, i);
    for (unsigned int j = 0; j < N; j++)
      if ( checkdep(ode, u, f0, i, j) )
	size++;
    
    // Compute total number of dependencies
    sum += size;

    // Allocate list
    list[i].init(size);

    // Set the dependencies
    unsigned int pos = 0;
    for (unsigned int j = 0; j < N; j++)
      if ( checkdep(ode, u, f0, i, j) )
	list[i](pos++) = j;

    // Update progress
    p = i;
  }

  dolfin_info("Automatically detected %d dependencies.", sum);
}
//-----------------------------------------------------------------------------
bool AutomaticSparsity::checkdep(ODE& ode, Vector& u, real f0, unsigned int i, unsigned int j)
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
AutomaticSparsity::Iterator::Iterator(unsigned int i, const AutomaticSparsity& sparsity)
  : GenericSparsity::Iterator(i), s(sparsity)
{
  pos = 0;

  if ( s.list[i].size() == 0 )
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
  if ( (pos + 1) == static_cast<unsigned int>(s.list[i].size()) )
    at_end = true;
  else
    pos++;

  return *this;
}
//-----------------------------------------------------------------------------
unsigned int AutomaticSparsity::Iterator::operator*() const
{
  return s.list[i](pos);
}
//-----------------------------------------------------------------------------
bool AutomaticSparsity::Iterator::end() const
{
  return at_end;
}
//-----------------------------------------------------------------------------
