// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Map.h>
#include <dolfin/Quadrature.h>
#include <dolfin/P1TriElement.h>
#include <dolfin/P1TetElement.h>
#include <dolfin/P1TriMap.h>
#include <dolfin/P1TetMap.h>
#include <dolfin/TriangleMidpointQuadrature.h>
#include <dolfin/TetrahedronMidpointQuadrature.h>
#include <dolfin/FiniteElementMethod.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FiniteElementMethod::FiniteElementMethod(Mesh::Type type, unsigned int noeq)
{
  // Create default finite element
  switch ( type ) {
  case Mesh::triangles:
    
    cout << "Using standard piecewise linears on triangles." << endl;
    
    element = new FiniteElement::Vector(noeq);
    for (unsigned int i = 0; i < noeq; ++i)
      (*element)(i) = new P1TriElement();
    
    map = new P1TriMap();
    quadrature = new TriangleMidpointQuadrature();
    
    break;
    
  case Mesh::tetrahedrons:
    
    cout << "Using standard piecewise linears on tetrahedrons." << endl;
    
    element = new FiniteElement::Vector(noeq);
    for (unsigned int i = 0; i < noeq; ++i)
      (*element)(i) = new P1TetElement();
    
    map = new P1TetMap();
    quadrature = new TetrahedronMidpointQuadrature();
    break;
    
  default:
    dolfin_error("No default method for this type of mesh.");
  }
}
//-----------------------------------------------------------------------------
FiniteElementMethod::FiniteElementMethod(FiniteElement::Vector& element,
					 Map& map, Quadrature& quadrature)
{
  this->element = &element;
  this->map = &map;
  this->quadrature = &quadrature;
  
  // User specifies method
  user = true;
}
//-----------------------------------------------------------------------------
FiniteElementMethod::~FiniteElementMethod()
{
  // Delete method data if not given by user
  if ( !user )
  {
    if ( element )
    {
      for (unsigned int i = 0; i < element->size(); ++i)
      {
	delete (*element)(i);
	(*element)(i) = 0;
      }
      delete element;
    }
    element = 0;
    
    if ( map )
      delete map;
    map = 0;
    
    if ( quadrature )
      delete quadrature;
    quadrature = 0;
  }
}
//-----------------------------------------------------------------------------
