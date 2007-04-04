// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-09
// Last changed: 2007-04-02

#include <dolfin/Vector.h>
#include <dolfin/P1tri.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/ConstantFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ConstantFunction::ConstantFunction(real value)
  : GenericFunction(),
    value(value), _mesh(0), mesh_local(false), size(1)
{
  cout << "Creating constant function" << endl;

  // Do nothing
}
//-----------------------------------------------------------------------------
ConstantFunction::ConstantFunction(const ConstantFunction& f)
  : GenericFunction(),
    value(f.value), _mesh(f._mesh), mesh_local(false), size(1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ConstantFunction::~ConstantFunction()
{
  // Delete mesh if local
  if ( mesh_local )
    delete _mesh;
}
//-----------------------------------------------------------------------------
real ConstantFunction::operator()(const Point& p, uint i)
{
  return value;
}
//-----------------------------------------------------------------------------
real ConstantFunction::operator() (const Vertex& vertex, uint i)
{
  return value;
}
//-----------------------------------------------------------------------------
void ConstantFunction::sub(uint i)
{
  // Do nothing (value same for all components anyway)
}
//-----------------------------------------------------------------------------
void ConstantFunction::interpolate(real coefficients[], Cell& cell,
				   AffineMap& map, FiniteElement& element)
{
  // Evaluate function at interpolation points
  for (uint i = 0; i < element.spacedim(); i++)
    coefficients[i] = value;
}
//-----------------------------------------------------------------------------
dolfin::uint ConstantFunction::vectordim() const
{
  dolfin_error("Vector dimension unknown for constant function.");
  return 0;
}
//-----------------------------------------------------------------------------
Vector& ConstantFunction::vector()
{
  dolfin_error("No vector associated with function (and none can be attached).");
  return *(new Vector()); // Code will not be reached, make compiler happy
}
//-----------------------------------------------------------------------------
Mesh& ConstantFunction::mesh()
{
  if ( !_mesh )
    dolfin_error("No mesh associated with function (try attaching one).");
  return *_mesh;
}
//-----------------------------------------------------------------------------
FiniteElement& ConstantFunction::element()
{
  dolfin_error("No finite element associated with function (an none can be attached).");
  return *(new P1tri()); // Code will not be reached, make compiler happy
}
//-----------------------------------------------------------------------------
void ConstantFunction::attach(Vector& x, bool local)
{
  dolfin_error("Cannot attach vectors to constant functions.");
}
//-----------------------------------------------------------------------------
void ConstantFunction::attach(Mesh& mesh, bool local)
{
  // Delete old mesh if local
  if ( mesh_local )
    delete _mesh;

  // Attach new mesh
  _mesh = &mesh;
  mesh_local = local;
}
//-----------------------------------------------------------------------------
void ConstantFunction::attach(FiniteElement& element, bool local)
{
  dolfin_error("Cannot attach finite elements to constant functions.");
}
//-----------------------------------------------------------------------------
void ConstantFunction::interpolate(real* coefficients,
                                   const ufc::cell& cell,
                                   const ufc::finite_element& finite_element)
{
  cout << "Interpolating ConstantFunction" << endl;

  // Evaluate each dof to get coefficients for nodal basis expansion
  for (uint i = 0; i < finite_element.space_dimension(); i++)
    coefficients[i] = finite_element.evaluate_dof(i, *this, cell);

  // Compute size of value (number of entries in tensor value)
  size = 1;
  for (uint i = 0; i < finite_element.value_rank(); i++)
    size *= finite_element.value_dimension(i);
}
//-----------------------------------------------------------------------------
void ConstantFunction::evaluate(real* values,
                                const real* coordinates,
                                const ufc::cell& cell) const
{
  // Set all values to the constant value
  for (uint i = 0; i < size; i++)
    values[i] = value;
}
//-----------------------------------------------------------------------------
