// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Mesh.h>
#include <dolfin/NewFiniteElement.h>
#include <dolfin/Vector.h>
#include <dolfin/NewFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewFunction::NewFunction()
  : Variable("u", "A function"), _x(0), _mesh(0), _element(0), t(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(Vector& x)
  : Variable("u", "A function"), _x(&x), _mesh(0), _element(0), t(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(Vector& x, Mesh& mesh)
  : Variable("u", "A function"), _x(&x), _mesh(&mesh), _element(0), t(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(Vector& x, Mesh& mesh, const NewFiniteElement& element)
  : Variable("u", "A function"), _x(&x), _mesh(&mesh), _element(&element), t(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewFunction::~NewFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewFunction::project(const Cell& cell, real c[]) const
{
  // Need to have an element to compute projection
  if ( !_element )
    dolfin_error("Function must be defined in terms of a finite element to compute projection.");

  // Two cases: either function is defined in terms of an element and
  // a list of dofs or the function is user-defined (_x is null)
  if ( _x )
  {
    // First case: function defined in terms of an element an a list
    // of dofs so we just need to pick the values

    if ( _mesh && _mesh != &cell.mesh() )
      dolfin_error("Function is defined on a different mesh.");

    real* values = _x->array();
    for (uint i = 0; i < _element->spacedim(); i++)
      c[i] = values[_element->dof(i, cell, cell.mesh())];
    _x->restore(values);
  }
  else
  {
    // Second case: function is user-defined so we need to compute the
    // projection to the given finite element space

    // FIXME: This is just a temporary fix for P1 elements

    if ( _element->rank() == 0 )
    {
      for (uint i = 0; i < _element->spacedim(); i++)
	c[i] = (*this)(_element->coord(i, cell, cell.mesh()));
    }
    else if ( _element->rank() == 1 )
    {
      const uint scalar_dim = _element->spacedim() / _element->tensordim(0);
      for (uint i = 0; i < _element->spacedim(); i++)
      {
	const uint component = i / scalar_dim;
	c[i] = (*this)(_element->coord(i, cell, cell.mesh()), component);
      }
    }
    else
      dolfin_error("Cannot handle tensor valued functions.");
  } 

  // FIXME: If we know that the values are stored element-by-element
  // FIXME: in x, then we can optimize by just calling
  // FIXME: element::dof() one time with i = 0.
}
//-----------------------------------------------------------------------------
real NewFunction::operator() (const Node& node) const
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
real NewFunction::operator() (const Node& node, uint i) const
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
real NewFunction::operator()(const Point& point) const
{
  dolfin_error("User-defined function evaluation not implemented.");
  return 0.0;
}
//-----------------------------------------------------------------------------
real NewFunction::operator()(const Point& point, uint i) const
{
  dolfin_error("User-defined function evaluation not implemented.");
  return 0.0;
}
//-----------------------------------------------------------------------------
Mesh& NewFunction::mesh()
{
  if ( !_mesh )
    dolfin_error("Mesh has not been specified.");

  return *_mesh;
}
//-----------------------------------------------------------------------------
const NewFiniteElement& NewFunction::element() const
{
  if ( !_element )
    dolfin_error("Finite element has not been specified.");

  return *_element;
}
//-----------------------------------------------------------------------------
real NewFunction::time() const
{
  return t;
}
//-----------------------------------------------------------------------------
void NewFunction::set(real time)
{
  t = time;
}
//-----------------------------------------------------------------------------
void NewFunction::set(const NewFiniteElement& element)
{
  if ( _element )
    dolfin_warning("Overriding previous choice of finite element.");
  _element = &element;
}
//-----------------------------------------------------------------------------
