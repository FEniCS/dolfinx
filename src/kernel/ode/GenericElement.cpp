// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/GenericElement.h>

using namespace dolfin;

// Initialise static data
Vector dolfin::GenericElement::f;

//-----------------------------------------------------------------------------
GenericElement::GenericElement(int q, int index, TimeSlab* timeslab)
{
  dolfin_assert(q >= 0);
  
  // Allocate the list of nodal values
  this->q = q;
  values = new real[q+1];

  dolfin_debug("foo");

  for(int i = 0; i < q+1; i++)
  {
    values[i] = 0;
  }

  values[0] = 1;

  this->index = index;
  this->timeslab = timeslab;

  // Increase size of the common vector to the maximum required size
  // among all elements. The vector is short (of length max(q+1)), but
  // a lot of extra storage would be required if each element stored
  // its own vector.
  
  if ( (q+1) > f.size() )
    f.init(q+1);
}
//-----------------------------------------------------------------------------
GenericElement::~GenericElement()
{
  dolfin_debug("foo");

  delete [] values;
}
//-----------------------------------------------------------------------------
int GenericElement::within(real t) const
{
  dolfin_assert(timeslab);
  return timeslab->within(t);
}
//-----------------------------------------------------------------------------
bool GenericElement::within(TimeSlab* timeslab) const
{
  return this->timeslab == timeslab;
}
//-----------------------------------------------------------------------------
real GenericElement::starttime() const
{
  dolfin_assert(timeslab);
  return timeslab->starttime();
}
//-----------------------------------------------------------------------------
real GenericElement::endtime() const
{
  dolfin_assert(timeslab);
  return timeslab->endtime();
}
//-----------------------------------------------------------------------------
real GenericElement::timestep() const
{
  dolfin_assert(timeslab);
  return timeslab->length();
}
//-----------------------------------------------------------------------------
