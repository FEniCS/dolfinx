// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_math.h>
#include <dolfin/cGqMethods.h>

using namespace dolfin;

/// Global table of cG(q) methods
cGqMethods dolfin::cG;

//-----------------------------------------------------------------------------
cGqMethods::cGqMethods()
{
  methods.init(DOLFIN_PARAMSIZE);
  methods.reset();
}
//-----------------------------------------------------------------------------
cGqMethods::~cGqMethods()
{
  for (int i = 0; i < methods.size(); i++) {
    if ( methods(i) )
      delete methods(i);
    methods(i) = 0;
  }
}
//-----------------------------------------------------------------------------
const cGqMethod& cGqMethods::operator() (int q) const
{
  dolfin_assert(q >= 1);
  dolfin_assert(q < methods.size());
  dolfin_assert(methods(q));

  return *methods(q);
}
//-----------------------------------------------------------------------------
void cGqMethods::init(int q)
{
  // Check if we need to increase the size of the list
  if ( q >= methods.size() )
    cG.methods.resize(max(q+1,2*methods.size()));
   
  // Check if the method has already been initialized
  if ( cG.methods(q) )
    return;

  // Initialize the method
  cG.methods(q) = new cGqMethod(q);
}
//-----------------------------------------------------------------------------
