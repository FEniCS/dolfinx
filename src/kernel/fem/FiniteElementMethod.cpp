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
    
    _element = new FiniteElement::Vector(noeq);
    for (unsigned int i = 0; i < noeq; ++i)
      (*_element)(i) = new P1TriElement();
    
    _map = new P1TriMap();
    _quadrature = new TriangleMidpointQuadrature();
    
    break;
    
  case Mesh::tetrahedrons:
    
    cout << "Using standard piecewise linears on tetrahedrons." << endl;
    
    _element = new FiniteElement::Vector(noeq);
    for (unsigned int i = 0; i < noeq; ++i)
      (*_element)(i) = new P1TetElement();
    
    _map = new P1TetMap();
    _quadrature = new TetrahedronMidpointQuadrature();
    break;
    
  default:
    dolfin_error("No default method for this type of mesh.");
  }
}
//-----------------------------------------------------------------------------
FiniteElementMethod::FiniteElementMethod(FiniteElement::Vector& element,
					 Map& map, Quadrature& quadrature)
{
  _element = &element;
  _map = &map;
  _quadrature = &quadrature;
  
  // User specifies method
  user = true;
}
//-----------------------------------------------------------------------------
FiniteElement::Vector& FiniteElementMethod::element()
{
  return *_element;
}
//-----------------------------------------------------------------------------
Map& FiniteElementMethod::map()
{
  return *_map;
}
//-----------------------------------------------------------------------------
Quadrature& FiniteElementMethod::quadrature()
{
  return *_quadrature;
}
//-----------------------------------------------------------------------------
FiniteElementMethod::~FiniteElementMethod()
{
  // Delete method data if not given by user
  if ( !user )
  {
    if ( _element )
    {
      for (unsigned int i = 0; i < _element->size(); ++i)
      {
	delete (*_element)(i);
	(*_element)(i) = 0;
      }
      delete _element;
    }
    _element = 0;
    
    if ( _map )
      delete _map;
    _map = 0;
    
    if ( _quadrature )
      delete _quadrature;
    _quadrature = 0;
  }
}
//-----------------------------------------------------------------------------
