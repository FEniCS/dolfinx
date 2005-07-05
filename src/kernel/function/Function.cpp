// Copyright (C) 2003-2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-11-28
// Last changed: 2005

#include <dolfin/Point.h>
#include <dolfin/Node.h>
#include <dolfin/Cell.h>
#include <dolfin/Mesh.h>
#include <dolfin/Vector.h>
#include <dolfin/AffineMap.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Function.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function()
  : Variable("u", "A function"), _x(0), _mesh(0), _element(0), t(0),
    dofs(0), components(0), points(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(Vector& x)
  : Variable("u", "A function"), _x(&x), _mesh(0), _element(0), t(0),
    dofs(0), components(0), points(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(Vector& x, Mesh& mesh)
  : Variable("u", "A function"), _x(&x), _mesh(&mesh), _element(0), t(0),
  dofs(0), components(0), points(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(Vector& x, Mesh& mesh, const FiniteElement& element)
  : Variable("u", "A function"), _x(&x), _mesh(&mesh), _element(&element), t(0),
    dofs(0), components(0), points(0)
{
  // Allocate temporary data used for interpolation
  dofs = new int[element.spacedim()];
  components = new uint[element.spacedim()];
  points = new Point[element.spacedim()];
}
//-----------------------------------------------------------------------------
Function::~Function()
{
  if ( dofs )
    delete [] dofs;

  if ( components )
    delete [] components;
  
  if ( points )
    delete [] points;
}
//-----------------------------------------------------------------------------
void Function::interpolate(real coefficients[], const AffineMap& map) const
{
  // Need to have an element to compute projection
  if ( !_element )
    dolfin_error("Function must be defined in terms of a finite element to compute projection.");

  // Two cases: either function is defined in terms of an element and
  // a list of dofs or the function is user-defined (_x is null)
  if ( _x )
  {
    // First case: function defined in terms of an element and a
    // vector of dofs so we just need to pick the values

    const Cell& cell = map.cell();
    const Mesh& mesh = cell.mesh();

    if ( _mesh && _mesh != &mesh )
      dolfin_error("Function is defined on a different mesh.");

    real* values = _x->array();
    _element->dofmap(dofs, cell, mesh);
    for (uint i = 0; i < _element->spacedim(); i++)
      coefficients[i] = values[dofs[i]];
    _x->restore(values);    
  }
  else
  {
    // Second case: function is user-defined so we need to compute the
    // interpolation onto the given finite element space
    
    _element->pointmap(points, components, map);
    if ( _element->rank() == 0 )
    {
      for (uint i = 0; i < _element->spacedim(); i++)
	coefficients[i] = (*this)(points[i]);
    }
    else
    {
      for (uint i = 0; i < _element->spacedim(); i++)
	coefficients[i] = (*this)(points[i], components[i]);
    }
  }
}
//-----------------------------------------------------------------------------
real Function::operator() (const Node& node) const
{
  if ( _x )
  {
    // FIXME: This is just a temporary fix for P1 elements
    return (*_x)(node.id());
  }
  else
  {
    return (*this)(node.coord());
  }
}
//-----------------------------------------------------------------------------
real Function::operator() (const Node& node, uint i) const
{
  if ( _x )
  {
    // FIXME: This is just a temporary fix for P1 elements

    dolfin_assert(_mesh);
    return (*_x)(i * _mesh->noNodes() + node.id());
  }
  else
  {
    return (*this)(node.coord(), i);
  }
}
//-----------------------------------------------------------------------------
real Function::operator()(const Point& point) const
{
  dolfin_error("Point evaluation has not been supplied by user-defined function.");
  return 0.0;
}
//-----------------------------------------------------------------------------
real Function::operator()(const Point& point, uint i) const
{
  dolfin_error("Point evaluation has not been supplied by user-defined function.");
  return 0.0;
}
//-----------------------------------------------------------------------------
Mesh& Function::mesh()
{
  if ( !_mesh )
    dolfin_error("Mesh has not been specified.");

  return *_mesh;
}
//-----------------------------------------------------------------------------
const FiniteElement& Function::element() const
{
  if ( !_element )
    dolfin_error("Finite element has not been specified.");

  return *_element;
}
//-----------------------------------------------------------------------------
real Function::time() const
{
  return t;
}
//-----------------------------------------------------------------------------
void Function::set(real time)
{
  t = time;
}
//-----------------------------------------------------------------------------
void Function::set(const FiniteElement& element)
{
  bool alloc = false;
  
  if ( _element )
  {
    dolfin_warning("Overriding previous choice of finite element.");

    if ( element.spacedim() != _element->spacedim() )
    {
      delete [] dofs;
      delete [] components;
      delete [] points;      
      alloc = true;
    }
  }
  else
    alloc = true;

  _element = &element;

  if ( alloc )
  {
    dofs = new int[element.spacedim()];
    components = new uint[element.spacedim()];
    points = new Point[element.spacedim()];
  }
}
//-----------------------------------------------------------------------------
