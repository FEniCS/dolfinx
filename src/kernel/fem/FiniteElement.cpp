// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-05-02
// Last changed: 2006-02-20

#include <string>

#include <dolfin/dolfin_log.h>
#include <dolfin/P1tri.h>
#include <dolfin/P2tri.h>
#include <dolfin/P3tri.h>
#include <dolfin/P4tri.h>
#include <dolfin/P5tri.h>
#include <dolfin/P1tet.h>
#include <dolfin/P2tet.h>
#include <dolfin/P3tet.h>
#include <dolfin/P4tet.h>
#include <dolfin/P5tet.h>
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
    return new P1tri();
  }
  else if ( repr == "[ Lagrange finite element of degree 2 on a triangle ]" )
  {
    dolfin_info("Creating finite element: %s.", repr.c_str());
    return new P2tri();
  }
  else if ( repr == "[ Lagrange finite element of degree 3 on a triangle ]" )
  {
    dolfin_info("Creating finite element: %s.", repr.c_str());
    return new P3tri();
  }
  else if ( repr == "[ Lagrange finite element of degree 4 on a triangle ]" )
  {
    dolfin_info("Creating finite element: %s.", repr.c_str());
    return new P4tri();
  }
  else if ( repr == "[ Lagrange finite element of degree 5 on a triangle ]" )
  {
    dolfin_info("Creating finite element: %s.", repr.c_str());
    return new P5tri();
  }
  else if ( repr == "[ Lagrange finite element of degree 1 on a tetrahedron ]" )
  {
    dolfin_info("Creating finite element: %s.", repr.c_str());
    return new P1tet();
  }
  else if ( repr == "[ Lagrange finite element of degree 2 on a tetrahedron ]" )
  {
    dolfin_info("Creating finite element: %s.", repr.c_str());
    return new P2tet();
  }
  else if ( repr == "[ Lagrange finite element of degree 3 on a tetrahedron ]" )
  {
    dolfin_info("Creating finite element: %s.", repr.c_str());
    return new P3tet();
  }
  else if ( repr == "[ Lagrange finite element of degree 4 on a tetrahedron ]" )
  {
    dolfin_info("Creating finite element: %s.", repr.c_str());
    return new P4tet();
  }
  else if ( repr == "[ Lagrange finite element of degree 5 on a tetrahedron ]" )
  {
    dolfin_info("Creating finite element: %s.", repr.c_str());
    return new P5tet();
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
