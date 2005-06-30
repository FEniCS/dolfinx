// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <dolfin/dolfin_log.h>
#include <dolfin/FiniteElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FiniteElement::~FiniteElement()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FiniteElement::disp() const
{
  dolfin_info("Finite element data:");
  dolfin_info("--------------------");
  dolfin_info("");
  dolfin_info("  space dimension = %d", spacedim());
  dolfin_info("  shape dimension = %d", shapedim());
  dolfin_info("      tensor rank = %d", rank());
  dolfin_info("");
}
//-----------------------------------------------------------------------------
