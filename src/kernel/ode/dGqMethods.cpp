// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_math.h>
#include <dolfin/dGqMethods.h>

using namespace dolfin;

/// Global table of dG(q) methods
dGqMethods dolfin::dG;

//-----------------------------------------------------------------------------
dGqMethods::dGqMethods()
{
  methods.init(DOLFIN_PARAMSIZE);
  methods = 0;
}
//-----------------------------------------------------------------------------
dGqMethods::~dGqMethods()
{
  for (int i = 0; i < methods.size(); i++) {
    if ( methods(i) )
      delete methods(i);
    methods(i) = 0;
  }
}
//-----------------------------------------------------------------------------
const dGqMethod& dGqMethods::operator() (int q) const
{
  dolfin_assert(q >= 1);
  dolfin_assert(q < methods.size());
  dolfin_assert(methods(q));

  return *methods(q);
}
//-----------------------------------------------------------------------------
void dGqMethods::init(int q)
{
  // Check if we need to increase the size of the list
  if ( q >= methods.size() )
    dG.methods.resize(max(q+1,2*methods.size()));
   
  // Check if the method has already been initialized
  if ( dG.methods(q) )
    return;

  // Initialize the method
  dG.methods(q) = new dGqMethod(q);
}
//-----------------------------------------------------------------------------
