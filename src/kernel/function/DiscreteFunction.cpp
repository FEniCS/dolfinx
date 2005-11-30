// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2005-11-30

#include <dolfin/dolfin_log.h>
#include <dolfin/Point.h>
#include <dolfin/Node.h>
#include <dolfin/Cell.h>
#include <dolfin/Mesh.h>
#include <dolfin/Vector.h>
#include <dolfin/AffineMap.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/DiscreteFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Vector& x)
  : _x(&x), _mesh(0), _element(0)
{
  // Mesh and element need to be specified later or are automatically
  // chosen during assembly.
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Vector& x, Mesh& mesh)
  : _x(&x), _mesh(&mesh), _element(0)
{
  // Element needs to be specified later or are automatically
  // chosen during assembly.
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Vector& x, Mesh& mesh, FiniteElement& element)
  : _x(&x), _mesh(&mesh), _element(&element)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(const DiscreteFunction& f)
  : _x(f._x), _mesh(f._mesh), _element(f._element)
{
  // Do nothing, just copy the values
}
//-----------------------------------------------------------------------------
DiscreteFunction::~DiscreteFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real DiscreteFunction::operator()(const Point& p, uint i)
{
  dolfin_error("Discrete functions cannot be evaluated at arbitrary points.");
  return 0.0;
}
//-----------------------------------------------------------------------------
real DiscreteFunction::operator() (const Node& node, uint i)
{
  dolfin_assert(_x && _mesh && _element);

  // Initialize local data (if not already initialized correctly)
  local.init(*_element);

  // Evaluate all components at given vertex and pick given component
  _element->vertexeval(local.values, node.id(), *_x, *_mesh);
  return local.values[i];
}
//-----------------------------------------------------------------------------
void DiscreteFunction::interpolate(real coefficients[], AffineMap& map,
				   FiniteElement& element)
{
  // Save mesh and element
  _mesh = &map.cell().mesh();
  _element = &element;
  
  // Initialize local data (if not already initialized correctly)
  local.init(*_element);
  
  // Get array of values (assumes uniprocessor case)
  real* xx = _x->array();
  
  // Compute mapping to global degrees of freedom
  _element->dofmap(local.dofs, map.cell(), *_mesh);

  // Pick values
  for (uint i = 0; i < _element->spacedim(); i++)
    coefficients[i] = xx[local.dofs[i]];

  // Restore array
  _x->restore(xx);    
}
//-----------------------------------------------------------------------------
dolfin::uint DiscreteFunction::vectordim() const
{
  dolfin_assert(_element);

  if ( _element->rank() == 0 )
  {
    return 1;
  }
  else if ( _element->rank() == 1 )
  {
    return _element->tensordim(0);
  }
  else
    dolfin_error("Cannot handle tensor-valued functions.");

  return 0;
}
//-----------------------------------------------------------------------------
Vector& DiscreteFunction::vector()
{
  dolfin_assert(_x);
  return *_x;
}
//-----------------------------------------------------------------------------
Mesh& DiscreteFunction::mesh()
{
  dolfin_assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
FiniteElement& DiscreteFunction::element()
{
  dolfin_assert(_element);
  return *_element;
}
//-----------------------------------------------------------------------------
void DiscreteFunction::attach(Vector& x)
{
  _x = &x;
}
//-----------------------------------------------------------------------------
void DiscreteFunction::attach(Mesh& mesh)
{
  _mesh = &mesh;
}
//-----------------------------------------------------------------------------
void DiscreteFunction::attach(FiniteElement& element)
{
  _element = &element;
}
//-----------------------------------------------------------------------------
