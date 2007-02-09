// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2005-05-17
// Last changed: 2006-12-06

#include <dolfin/Vertex.h>
#include <dolfin/Cell.h>
#include <dolfin/Edge.h>
#include <dolfin/Face.h>
#include <dolfin/FEBasis.h>
#include <dolfin/SpecialFunctions.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FEBasis::FEBasis() : functions(0)
{
}
//-----------------------------------------------------------------------------
FEBasis::~FEBasis()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool FEBasis::construct(FiniteElement& element)
{
  spec = element.spec();
  std::string repr = spec.repr();

  if ( repr == "[ Lagrange finite element of degree 1 on a triangle ]" )
  {
    dolfin_info("Constructing basis for: %s.", repr.c_str());
    
    functions.resize(3);
    functions[0] = new ScalarLagrange(0);
    functions[1] = new ScalarLagrange(1);
    functions[2] = new ScalarLagrange(2);
  }
  else if ( repr == "[ Discontinuous Lagrange finite element of degree 0 on a triangle ]" )
  {
    dolfin_info("Constructing basis for: %s.", repr.c_str());
    
    functions.resize(1);
    functions[0] = new ScalarDiscontinousLagrange(0);
  }
  else
  {
    dolfin_info("Unable to construct basis functions for: %s.", repr.c_str());

    return false;
  }
  
  return true;
}
//-----------------------------------------------------------------------------
real FEBasis::evalPhysical(Function& f, Point& p, NewAffineMap& map,
			   dolfin::uint i)
{
  Point pref = map.mapinv(p.x(), p.y(), p.z());

  return f(pref, i);
}
//-----------------------------------------------------------------------------
