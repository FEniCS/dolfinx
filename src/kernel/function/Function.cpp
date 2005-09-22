// Copyright (C) 2003-2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005
//
// First added:  2003-11-28
// Last changed: 2005-09-20

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
  : Variable("u", "A function"), TimeDependent(), 
    _x(0), _mesh(0), _element(0), _cell(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(Vector& x)
  : Variable("u", "A function"), TimeDependent(),
    _x(&x), _mesh(0), _element(0), _cell(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(Vector& x, Mesh& mesh)
  : Variable("u", "A function"), TimeDependent(),
    _x(&x), _mesh(&mesh), _element(0), _cell(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(Vector& x, Mesh& mesh, const FiniteElement& element)
  : Variable("u", "A function"), TimeDependent(),
    _x(&x), _mesh(&mesh), _element(&element), _cell(0)
{
  // Allocate local data
  local.init(element);
}
//-----------------------------------------------------------------------------
Function::~Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Function::interpolate(real coefficients[], const AffineMap& map)
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

    real* xvals = _x->array();
    _element->dofmap(local.dofs, cell, mesh);
    for (uint i = 0; i < _element->spacedim(); i++)
      coefficients[i] = xvals[local.dofs[i]];
    _x->restore(xvals);    
  }
  else
  {
    // Second case: function is user-defined so we need to compute the
    // interpolation onto the given finite element space
    
    const Cell& cell = map.cell();
    _cell = &cell;

    _element->pointmap(local.points, local.components, map);
    if ( _element->rank() == 0 )
    {
      for (uint i = 0; i < _element->spacedim(); i++)
	coefficients[i] = (*this)(local.points[i]);
    }
    else
    {
      for (uint i = 0; i < _element->spacedim(); i++)
	coefficients[i] = (*this)(local.points[i], local.components[i]);
    }
  }
}
//-----------------------------------------------------------------------------
real Function::operator() (const Node& node) const
{
  if ( _x )
  {
    // Need to have a mesh to evaluate function
    if ( !_mesh )
      dolfin_error("Mesh not specified for function.");

    // Need to have an element to evaluate function
    if ( !_element )
      dolfin_error("Finite element not specified for function.");

    // Evaluate all components at given vertex and pick first component
    _element->vertexeval(local.values, node.id(), *_x, *_mesh);
    return local.values[0];
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
    // Need to have a mesh to evaluate function
    if ( !_mesh )
      dolfin_error("Mesh not specified for function.");

    // Need to have an element to evaluate function
    if ( !_element )
      dolfin_error("Finite element not specified for function.");

    // Evaluate all components at given vertex and pick given component
    _element->vertexeval(local.values, node.id(), *_x, *_mesh);
    return local.values[i];
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
Vector& Function::vector()
{
  if ( !_x )
    dolfin_error("Vector has not been specified.");
  
  return *_x;
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
void Function::set(const FiniteElement& element)
{
  // Give a warning if element has previously been specified
  if ( _element )
    dolfin_warning("Overriding previous choice of finite element.");
  
  // Allocate local data
  local.init(element);

  // Save given element
  _element = &element;
}
//-----------------------------------------------------------------------------
Function::LocalData::LocalData() : dofs(0), components(0), points(0), values(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::LocalData::~LocalData()
{
  // Clear data if initialized
  clear();
}
//-----------------------------------------------------------------------------
void Function::LocalData::init(const FiniteElement& element)
{
  // Clear data if initialized
  clear();

  // Initialize local degrees of freedom
  dofs = new int[element.spacedim()];

  // Initialize local components
  components = new uint[element.spacedim()];

  // Initialize local nodal points
  points = new Point[element.spacedim()];

  // Initialize local vertex values
  if ( element.rank() == 0 )
    values = new real[1];
  else
    values = new real[element.tensordim(0)];
}
//-----------------------------------------------------------------------------
void Function::LocalData::clear()
{
  if ( dofs )
    delete [] dofs;
  dofs = 0;
  
  if ( components )
    delete [] components;
  components = 0;
  
  if ( points )
    delete [] points;
  points = 0;
  
  if ( values )
    delete [] values;
  values = 0;
}
//-----------------------------------------------------------------------------
