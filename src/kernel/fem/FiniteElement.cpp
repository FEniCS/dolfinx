// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-05-02
// Last changed: 2006-02-20

#include <string>

#include <dolfin/dolfin_log.h>
#include <dolfin/P1Tri.h>
#include <dolfin/P1Tet.h>
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
FiniteElement* FiniteElement::makeElement(const FiniteElementSpec& spec)
{
  // Find the correct element from the string representation
  std::string repr = spec.repr();
  if ( repr == "[ Lagrange finite element of degree 1 on a triangle ]" )
  {
    dolfin_info("Creating finite element: %s.", repr.c_str());
    return new P1Tri();
  }
  else if ( repr == "[ Lagrange finite element of degree 1 on a tetrahedron ]" )
  {
    return new P1Tet();
    dolfin_info("Creating finite element: %s.", repr.c_str());
  }

  dolfin_warning1("%s", repr.c_str());
  dolfin_warning("Unable to create specified finite element, no matching precompiled element found.");

  return 0;
}
//-----------------------------------------------------------------------------
FiniteElement* FiniteElement::makeElement(std::string type, std::string shape,
					  uint degree, uint vectordim)
{
  FiniteElementSpec spec(type, shape, degree, vectordim);
  return makeElement(spec);
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
