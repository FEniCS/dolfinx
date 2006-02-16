// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-09
// Last changed: 2006-02-16

#include <dolfin/Vector.h>
#include <dolfin/P1Tri.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/ConstantFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ConstantFunction::ConstantFunction(real value)
  : GenericFunction(),
    value(value), _mesh(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ConstantFunction::ConstantFunction(const ConstantFunction& f)
  : GenericFunction(),
    value(f.value), _mesh(f._mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ConstantFunction::~ConstantFunction()
{
  // Do nothing
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
void ConstantFunction::interpolate(real coefficients[],
				   AffineMap& map,
				   FiniteElement& element)
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
  return *(new P1Tri()); // Code will not be reached, make compiler happy
}
//-----------------------------------------------------------------------------
void ConstantFunction::attach(Vector& x)
{
  dolfin_error("Cannot attach vectors to constant functions.");
}
//-----------------------------------------------------------------------------
void ConstantFunction::attach(Mesh& mesh)
{
  _mesh = &mesh;
}
//-----------------------------------------------------------------------------
void ConstantFunction::attach(FiniteElement& element)
{
  dolfin_error("Cannot attach finite elements to constant functions.");
}
//-----------------------------------------------------------------------------
void ConstantFunction::attach(std::string element)
{
  dolfin_error("Cannot attach finite elements to constant functions.");
}
//-----------------------------------------------------------------------------
