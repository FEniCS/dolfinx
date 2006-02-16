// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-05-02
// Last changed: 2006-02-16

#include <sstream>

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
FiniteElement* FiniteElement::makeElement(std::string type, std::string shape,
					  uint degree, uint vectordim)
{
  /*
  if ( type == "Lagrange" )
  {
    



  }
  else
  {

  }
  */

  dolfin_warning1("%s", repr(type, shape, degree, vectordim).c_str());
  dolfin_warning("Unable to create specified finite element, no matching precompiled element found.");

  return 0;
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
std::string FiniteElement::repr(std::string type, std::string shape,
				uint degree, uint vectordim)
{
  std::ostringstream stream;
  
  if ( vectordim > 1 )
  {
    stream << "[ " << type << " finite element of degree " << degree
	   << " on a " << shape << " with " << vectordim << " ]";
  }
  else
  {
    stream << type << " finite element of degree " << degree
	   << " on a " << shape;
  }

  return stream.str();
}
//-----------------------------------------------------------------------------
