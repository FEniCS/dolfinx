// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ElementData.h>
#include <dolfin/Element.h>
#include <dolfin/ODEFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ODEFunction::ODEFunction(ElementData& elmdata) : elmdata(elmdata)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ODEFunction::~ODEFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real ODEFunction::operator() (unsigned int i, real t) const
{
  // Get element
  Element* element = elmdata.element(i,t);

  // Check if we got the element
  if ( !element )
    dolfin_error("Requested value not available.");

  // Evaluate element at given time
  return element->value(t);
}
//-----------------------------------------------------------------------------
